"""This module loads a pre-trained ANN to predict job resource requirements for HST.
# 1 - load job metadata inputs from text file in s3
# 2 - encode strings as int/float values in numpy array
# 3 - load models and generate predictions
# 4 - return preds as json to parent lambda function

Predict Resource Allocation requirements for memory (GB) and max execution `kill time` / `wallclock` (seconds) using three pre-trained neural networks.

MEMORY BIN: classifier outputs probabilities for each of the four bins ("target classes"). The class with the highest probability score is considered the final predicted outcome (y). This prediction variable represents which of the 4 possible memory bins is most likely to meet the minimum required needs for processing an HST dataset (ipppssoot) successfully according to the given inputs (x).

Memory Bin Sizes (target class "y"):
0: < 2GB
1: 2-8GB
2: 8-16GB
3: >16GB

WALLCLOCK REGRESSION: regression generates estimate for specific number of seconds needed to process the dataset using the same input data. This number is then tripled in Calcloud for the sake of creating an extra buffer of overhead in order to prevent larger jobs from being killed unnecessarily.

MEMORY REGRESSION: A third regression model is used to estimate the actual value of memory needed for the job. This is mainly for the purpose of logging/future analysis and is not currently being used for allocating memory in calcloud jobs.
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
import json


def get_model(model_path):
    """Loads pretrained Keras functional model"""
    model = tf.keras.models.load_model(model_path)
    return model


def read_inputs(
    n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr
):
    x_features = {
        "n_files": n_files,
        "total_mb": total_mb,
        "drizcorr": drizcorr,
        "pctecorr": pctecorr,
        "crsplit": crsplit,
        "subarray": subarray,
        "detector": detector,
        "dtype": dtype,
        "instr": instr,
    }
    print(x_features)
    return x_features


def classifier(model, data):
    """Returns class prediction"""
    pred_proba = model.predict(data)
    pred = int(np.argmax(pred_proba, axis=-1))
    return pred, pred_proba


def regressor(model, data):
    """Returns Regression model prediction"""
    pred = model.predict(data)
    return pred


class Preprocess:
    def __init__(self, x_features, pt_file):
        self.x_features = x_features
        self.pt_file = pt_file
        self.inputs = None
        self.lambdas = None
        self.f_mean = None
        self.f_sigma = None
        self.s_mean = None
        self.s_sigma = None

    def scrub_keys(self):
        x = self.x_features
        n_files = x["n_files"]
        total_mb = x["total_mb"]
        detector = x["detector"]
        subarray = x["subarray"]
        drizcorr = x["drizcorr"]
        pctecorr = x["pctecorr"]
        crsplit = x["crsplit"]
        dtype = x["dtype"]
        instr = x["instr"]

        inputs = np.array(
            [
                n_files,
                total_mb,
                drizcorr,
                pctecorr,
                crsplit,
                subarray,
                detector,
                dtype,
                instr,
            ]
        )
        return inputs

    def load_pt_data(self):
        with open(self.pt_file, "r") as j:
            pt_data = json.load(j)
        
        self.lambdas = np.array([pt_data["f_lambda"], pt_data["s_lambda"]])
        self.f_mean = pt_data["f_mean"]
        self.f_sigma = pt_data["f_sigma"]
        self.s_mean = pt_data["s_mean"]
        self.s_sigma = pt_data["s_sigma"]

        return self

    def transformer(self):
        """applies yeo-johnson power transform to first two indices of array (n_files, total_mb) using lambdas, mean and standard deviation calculated for each variable prior to model training.

        Returns: X inputs as 2D-array for generating predictions
        """
        X = self.inputs
        n_files = X[0]
        total_mb = X[1]
        # apply power transformer normalization to continuous vars
        x = np.array([[n_files], [total_mb]]).reshape(1, -1)
        pt = PowerTransformer(standardize=False)
        pt.lambdas_ = self.lambdas
        xt = pt.transform(x)
        # normalization (zero mean, unit variance)
        x_files = np.round(((xt[0, 0] - self.f_mean) / self.f_sigma), 5)
        x_size = np.round(((xt[0, 1] - self.s_mean) / self.s_sigma), 5)
        X = np.array(
            [x_files, x_size, X[2], X[3], X[4], X[5], X[6], X[7], X[8]]
        ).reshape(1, -1)
        print(X)
        return X


def make_preds(x_features, NN=None):
    global clf
    global mem_reg
    if NN is None:
        clf = get_model("./models/mem_clf")
        mem_reg = get_model("./models/mem_reg/")
        wall_reg = get_model("./models/wall_reg/")
    else:
        clf = NN["clf"]
        mem_reg = NN["mem_reg"]
        wall_reg = NN["wall_reg"]
    prep = Preprocess(x_features, "./models/pt_transform")
    prep.inputs = prep.scrub_keys()
    prep.load_pt_data()
    X = prep.transformer()
    # Predict Memory Allocation (bin and value preds)
    membin, pred_proba = classifier(clf, X)
    P = pred_proba[0]
    p0, p1, p2, p3 = P[0], P[1], P[2], P[3]
    memval = np.round(float(regressor(mem_reg, X)), 2)
    print(f"\n*** BIN PRED: {membin}\n***PROBABILITIES: {P}\n***MEMORY:{memval}")
    # Predict Wallclock Allocation (execution time in seconds)
    clocktime = int(regressor(wall_reg, X))
    memory_predictions = [membin, memval, clocktime, p0, p1, p2, p3]
    return memory_predictions


def relu_activation(layer_num, input_arr, neuron):
    # get weights for each neuron
    w_dense = np.array(clf.layers[layer_num].weights[0])
    weights = [w[neuron] for w in w_dense]
    # get bias values
    b = np.array(clf.layers[layer_num].bias)[neuron]
    wxb = np.sum(weights * input_arr) + b
    n = np.max([0, wxb])
    return n


def layer_neurons(layer_num, input_arr):
    w_dense = np.array(clf.layers[layer_num].weights[0])
    n_neurons = w_dense.shape[1]
    neuron_values = []
    for n in list(range(n_neurons)):
        s = relu_activation(layer_num, input_arr, n)
        neuron_values.append(s)
    print(f"\n*** RELU ACTIVATION: \n{neuron_values}")
    return np.array(neuron_values)


def softmax_activation(out):
    """
    Softmax function to convert values from output layer into probabilities btw 0 and 1.
    Note: tensorflow has a function for this but I wanted to learn/check the math myself
    **args:
    `out`: array of neuron values from output layer
    out = array([6.04094268, 0.        , 1.08320938, 1.1343926 ])
    SOFTMAX: array([5.23e+04, 1.00e+00, 1.09+00, 1.15e+00])
    Y = [0.999, 1.911e-05, 2.08e-05, 2.205e-05]
    """
    e = np.zeros(shape=(4))
    for i, E in enumerate(out):
        e[i] = E ** E
    print(f"\n*** SOFTMAX: {e}")
    denom = e[0] + e[1] + e[2] + e[3]
    with np.printoptions(floatmode="maxprec"):
        Y = np.array([e[0] / denom, e[1] / denom, e[2] / denom, e[3] / denom])
    print(f"\n*** Y = \n{Y}")
    print(f"\n*** CLASS PREDICTION: {np.argmax(Y)}")
    return Y


def calculate_neurons(input_arr):
    n1 = layer_neurons(1, input_arr)
    n2 = layer_neurons(2, n1)
    n3 = layer_neurons(3, n2)
    n4 = layer_neurons(4, n3)
    n5 = layer_neurons(5, n4)
    n6 = layer_neurons(6, n5)
    out = layer_neurons(7, n6)
    y = softmax_activation(out)
    neurons = [n1, n2, n3, n4, n5, n6, y]
    return neurons
