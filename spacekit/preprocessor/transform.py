import json
import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf


class Transformer:
    def __init__(
        self, data, transformer=PowerTransformer(standardize=False), cols=None
    ):
        self.data = data  # cal: data = x_features
        self.transformer = transformer
        self.cols = cols
        self.matrix = self.frame_to_matrix()
        self.matrix_cont = None
        self.matrix_cat = None
        self.matrix_norm = None
        self.idx = self.data.index
        self.data_cont = None
        self.data_cat = None
        self.data_norm = None
        self.tx_file = None
        self.tx_data = None
        self.lambdas = None

    def load_transformer_data(self):
        if self.tx_file is not None:
            with open(self.tx_file, "r") as j:
                self.tx_data = json.load(j)
            return self.tx_data
        else:
            return None

    def frame_to_matrix(self):
        if type(self.data) == pd.DataFrame:
            self.matrix = self.data.values
        return self.matrix

    def power_matrix(self):
        try:
            nrows = self.matrix_cont.shape[0]
            ncols = self.matrix_cont.shape[1]
            self.transformer.fit(self.matrix_cont)
            self.transformer.lambdas_ = self.tx_data["lambdas"]
            input_matrix = self.transformer.transform(self.matrix_cont)
            normalized = np.empty((nrows, ncols))
            for i in range(ncols):
                v = input_matrix[:, i]
                m = self.tx_data["mu"][i]
                s = self.tx_data["sigma"][i]
                x = (v - m) / s
                normalized[:, i] = x
            self.matrix_norm = np.concatenate((normalized, self.matrix_cat), axis=1)
            return self.matrix_norm
        except Exception as e:
            print(
                "Err: Continuous/Categorical matrices (`matrix_cont`, `matrix_cat`) need to be instantiated."
            )
            print(e)
            return None

    def power_frame(self):
        # data_cont = data[cols]
        self.transformer.fit(self.data_cont)
        input_matrix = self.transformer.transform(self.data_cont)
        self.lambdas = self.transformer.lambdas_
        normalized = np.empty((len(self.data), len(self.cols)))
        mu, sig = [], []
        for i in range(len(self.cols)):
            v = input_matrix[:, i]
            m, s = np.mean(v), np.std(v)
            x = (v - m) / s
            normalized[:, i] = x
            mu.append(m)
            sig.append(s)
        self.tx_data = {
            "lambdas": self.lambdas,
            "mu": np.array(mu),
            "sigma": np.array(sig),
        }
        newcols = [c + "_scl" for c in self.cols]
        df_norm = pd.DataFrame(normalized, index=self.idx, columns=newcols)
        self.data_norm = df_norm.join(self.data_cat, how="left")
        return self


class SvmX(Transformer):
    def __init__(self, data, tx_file=None):
        super.__init__(data)
        self.tx_file = tx_file
        self.cols = [
            "numexp",
            "rms_ra",
            "rms_dec",
            "nmatches",
            "point",
            "segment",
            "gaia",
        ]
        self.matrix_cont = self.data[:, :7]  # continuous
        self.matrix_cat = self.data[:, -3:]  # categorical
        self.data_cont = self.data[self.cols]
        self.data_cat = self.data.drop(self.cols, axis=1, inplace=False)


class CalX(Transformer):
    def __init__(self, data, tx_file=None):
        super().__init__(data)
        self.tx_file = tx_file
        self.cols = ["n_files", "total_mb"]
        self.tx_data = self.load_transformer_data()
        self.X = self.powerX()

    def transform(self):
        if self.tx_data is not None:
            self.inputs = self.scrub_keys()
            self.lambdas = np.array(
                [self.tx_data["f_lambda"], self.tx_data["s_lambda"]]
            )
            self.f_mean = self.tx_data["f_mean"]
            self.f_sigma = self.tx_data["f_sigma"]
            self.s_mean = self.tx_data["s_mean"]
            self.s_sigma = self.tx_data["s_sigma"]
            return self

    def scrub_keys(self):
        x = self.data
        self.inputs = np.array(
            [
                x["n_files"],
                x["total_mb"],
                x["drizcorr"],
                x["pctecorr"],
                x["crsplit"],
                x["subarray"],
                x["detector"],
                x["dtype"],
                x["instr"],
            ]
        )
        return self.inputs

    def powerX(self):
        """applies yeo-johnson power transform to first two indices of array (n_files, total_mb) using lambdas, mean and standard deviation pre-calculated for each variable (loads dict from json file).

        Returns: X inputs as 2D-array for generating predictions
        """
        if self.inputs is None:
            return None
        else:
            X = self.inputs
            n_files = X[0]
            total_mb = X[1]
            # apply power transformer normalization to continuous vars
            x = np.array([[n_files], [total_mb]]).reshape(1, -1)
            self.transformer.lambdas_ = self.lambdas
            xt = self.transformer.transform(x)
            # normalization (zero mean, unit variance)
            x_files = np.round(((xt[0, 0] - self.f_mean) / self.f_sigma), 5)
            x_size = np.round(((xt[0, 1] - self.s_mean) / self.s_sigma), 5)
            self.X = np.array(
                [x_files, x_size, X[2], X[3], X[4], X[5], X[6], X[7], X[8]]
            ).reshape(1, -1)
            print(self.X)
            return self.X


# TODO: update code elsewhere to use class method versions instead of static functions


def apply_power_transform(
    data, cols=["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"]
):
    data_cont = data[cols]
    idx = data_cont.index
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    input_matrix = pt.transform(data_cont)
    lambdas = pt.lambdas_
    normalized = np.empty((len(data), len(cols)))
    mu, sig = [], []
    for i in range(len(cols)):
        v = input_matrix[:, i]
        m, s = np.mean(v), np.std(v)
        x = (v - m) / s
        normalized[:, i] = x
        mu.append(m)
        sig.append(s)
    pt_dict = {"lambdas": lambdas, "mu": np.array(mu), "sigma": np.array(sig)}
    newcols = [c + "_scl" for c in cols]
    df_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
    df = data.drop(cols, axis=1, inplace=False)
    df = df_norm.join(df, how="left")
    return df, pt_dict


def power_transform_matrix(data, pt_data):
    if type(data) == pd.DataFrame:
        data = data.values
    data_cont = data[:, :7]
    data_cat = data[:, -3:]
    nrows = data_cont.shape[0]
    ncols = data_cont.shape[1]
    pt = PowerTransformer(standardize=False)
    pt.fit(data_cont)
    pt.lambdas_ = pt_data["lambdas"]
    input_matrix = pt.transform(data_cont)
    normalized = np.empty((nrows, ncols))
    for i in range(data_cont.shape[1]):
        v = input_matrix[:, i]
        m = pt_data["mu"][i]
        s = pt_data["sigma"][i]
        x = (v - m) / s
        normalized[:, i] = x
    data_norm = np.concatenate((normalized, data_cat), axis=1)
    return data_norm


def update_power_transform(df):
    pt = PowerTransformer(standardize=False)
    df_cont = df[["n_files", "total_mb"]]
    pt.fit(df_cont)
    input_matrix = pt.transform(df_cont)
    # FILES (n_files)
    f_mean = np.mean(input_matrix[:, 0])
    f_sigma = np.std(input_matrix[:, 0])
    # SIZE (total_mb)
    s_mean = np.mean(input_matrix[:, 1])
    s_sigma = np.std(input_matrix[:, 1])
    files = input_matrix[:, 0]
    size = input_matrix[:, 1]
    x_files = (files - f_mean) / f_sigma
    x_size = (size - s_mean) / s_sigma
    normalized = np.stack([x_files, x_size], axis=1)
    idx = df_cont.index
    df_norm = pd.DataFrame(normalized, index=idx, columns=["x_files", "x_size"])
    df["x_files"] = df_norm["x_files"]
    df["x_size"] = df_norm["x_size"]
    lambdas = pt.lambdas_
    pt_transform = {
        "f_lambda": lambdas[0],
        "s_lambda": lambdas[1],
        "f_mean": f_mean,
        "f_sigma": f_sigma,
        "s_mean": s_mean,
        "s_sigma": s_sigma,
    }
    print(pt_transform)
    return df, pt_transform


def make_tensors(X_train, y_train, X_test, y_test):
    """Convert Arrays to Tensors"""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, y_train, X_test, y_test


def array_to_tensor(arr):
    """Convert Arrays to Tensors"""
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    return tensor


def tensor_to_array(tensor, reshape=False):
    if reshape:
        arr = np.asarray(tensor).reshape(-1, 1)
    else:
        arr = np.asarray(tensor)
    return arr


def tensors_to_arrays(X_train, y_train, X_test, y_test):
    """Converts tensors into arrays, which is necessary for certain computations.
    Returns:
        Arrays (4): X_train, y_train, X_test, y_test (arrays)
    """
    X_train = tensor_to_array(X_train)
    y_train = tensor_to_array(y_train, reshape=True)
    X_test = tensor_to_array(X_test)
    y_test = tensor_to_array(y_test, reshape=True)
    return X_train, y_train, X_test, y_test


def make_arrays(X_train, y_train, X_test, y_test):
    X_train = X_train.values
    y_train = y_train.values.reshape(-1, 1)
    X_test = X_test.values
    y_test = y_test.values.reshape(-1, 1)
    return X_train, y_train, X_test, y_test


def hypersonic_pliers(path_to_train, path_to_test):

    """
    Using Numpy to extract data into 1-dimensional arrays
    separate target classes (y) for training and test data
    assumes y (target) is first column in dataframe

    #TODO: option to pass in column index loc for `y` if not default (0)
    #TODO: option for `y` to already be 0 or 1 (don't subtract 1)
    #TODO: allow option for `y` to be categorical (convert via binarizer)
    """

    Train = np.loadtxt(path_to_train, skiprows=1, delimiter=",")
    X_train = Train[:, 1:]
    y_train = Train[:, 0, np.newaxis] - 1.0

    Test = np.loadtxt(path_to_test, skiprows=1, delimiter=",")
    X_test = Test[:, 1:]
    y_test = Test[:, 0, np.newaxis] - 1.0

    del Train, Test
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    return X_train, X_test, y_train, y_test


def thermo_fusion_chisel(matrix1, matrix2=None):
    """
    Scales each array of a matrix to zero mean and unit variance.
    returns matrix/matrices of same shape as input but scaled
    matrix2 is optional - useful if data was already train-test split
    example: matrix1=X_train, matrix2=X_test

    """

    matrix1 = (matrix1 - np.mean(matrix1, axis=1).reshape(-1, 1)) / np.std(
        matrix1, axis=1
    ).reshape(-1, 1)

    print("Mean: ", matrix1[0].mean())
    print("Variance: ", matrix1[0].std())

    if matrix2 is not None:
        matrix2 = (matrix2 - np.mean(matrix2, axis=1).reshape(-1, 1)) / np.std(
            matrix2, axis=1
        ).reshape(-1, 1)

        print("Mean: ", matrix2[0].mean())
        print("Variance: ", matrix2[0].std())
        return matrix1, matrix2
    else:
        return matrix1


def babel_fish_dispenser(matrix1, matrix2=None, step_size=None, axis=2):
    """
    Adds an input corresponding to the running average over a set number
    of time steps. This helps the neural network to ignore high frequency
    noise by passing in a uniform 1-D filter and stacking the arrays.

    **ARGS
    step_size: integer, # timesteps for 1D filter. defaults to 200
    axis: which axis to stack the arrays

    ex:
    babel_fish_dispenser(matrix1=X_train, matrix2=X_test, step_size=200)
    """
    if step_size is None:
        step_size = 200

    # calc input for flux signal rolling avgs
    filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
    # store in array and stack on 2nd axis for each obs of X data
    matrix1 = np.stack([matrix1, filter1], axis=2)

    if matrix2 is not None:
        filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
        matrix2 = np.stack([matrix2, filter2], axis=2)
        print(matrix1.shape, matrix2.shape)
        return matrix1, matrix2
    else:
        print(matrix1.shape)
        return matrix1


def fast_fourier(matrix, bins):
    """
    takes in array and rotates #bins to the left as a fourier transform
    returns vector of length equal to input array
    """

    shape = matrix.shape
    fourier_matrix = np.zeros(shape, dtype=float)

    for row in matrix:
        signal = np.asarray(row)
        frequency = np.arange(signal.size / 2 + 1, dtype=np.float)
        phase = np.exp(
            complex(0.0, (2.0 * np.pi)) * frequency * bins / float(signal.size)
        )
        ft = np.fft.irfft(phase * np.fft.rfft(signal))
        fourier_matrix += ft

    return fourier_matrix
