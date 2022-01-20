import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import uniform_filter1d
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf


class Transformer:
    def __init__(
        self, data, cols=[], tx_data=None, tx_file=None, save_tx=True, output_path=None
    ):
        """Instantiates a Transformer class object. Unless the `cols` attribute is empty, it will automatically instantiate some of the other attributes needed to transform the data. Using the Transformer subclasses instead is recommended (this class is mainly used as an object with general methods to load or save the transform data as well as instantiate some of the initial attributes).

        Parameters
        ----------
        data : dataframe or numpy.ndarray
            input data containing continuous feature vectors to be transformed (may also contain vectors or columns of categorical and other datatypes as well).
        transformer : class, optional
            transform class to use (e.g. from scikit-learn), by default PowerTransformer(standardize=False)
        cols : list, optional
            column names or array index values of feature vectors to be transformed (i.e. continuous datatype features), by default []
        tx_file : string, optional
            path to saved transformer metadata, by default None
        save_tx : bool, optional
            save the transformer metadata as json file on local disk, by default True
        output_path : string, optional
            where to save the transformer metadata, by default None (current working directory)
        """
        self.data = data
        self.cols = cols
        self.tx_file = tx_file
        self.save_tx = save_tx
        self.output_path = output_path
        self.tx_data = self.load_transformer_data(tx=tx_data)
        self.continuous = self.continuous_data()
        self.categorical = self.categorical_data()

    def load_transformer_data(self, tx=None):
        """Loads saved transformer metadata from a dictionary or a json file on local disk.

        Returns
        -------
        dictionary
            transform metadata used for applying transformations on new data inputs
        """
        if tx:
            self.tx_data = tx
        elif self.tx_file is not None:
            with open(self.tx_file, "r") as j:
                self.tx_data = json.load(j)
            return self.tx_data
        else:
            return None

    def save_transformer_data(self, tx=None):
        """Save the transform metadata to a json file on local disk. Typical use-case is when you need to transform new inputs prior to generating a prediction but don't have access to the original dataset used to train the model.

        Parameters
        ----------
        tx : dictionary
            statistical metadata calculated when applying a transform to the training dataset; for PowerTransform this consists of lambdas, means and standard deviations for each continuous feature vector of the dataset.

        Returns
        -------
        string
            path where json file is saved on disk
        """
        if self.output_path is None:
            self.output_path = os.getcwd()
        else:
            os.makedirs(self.output_path, exist_ok=True)
        self.tx_file = f"{self.output_path}/tx_data"
        with open(self.tx_file, "w") as j:
            if tx is None:
                json.dump(self.tx_data, j)
            else:
                json.dump(tx, j)
        print("TX data saved as json file: ", self.tx_file)
        return self.tx_file

    def continuous_data(self):
        """Store continuous feature vectors in a variable using the column names (or axis index if using numpy arrays) from `cols` attribute.

        Returns
        -------
        dataframe or ndarray
            continuous feature vectors (as determined by `cols` attribute)
        """
        if self.cols is None:
            print("`cols` attribute not instantiated.")
            return None
        if type(self.data) == pd.DataFrame:
            return self.data[self.cols]
        elif type(self.data) == np.ndarray:
            return self.data[:, self.cols]

    def categorical_data(self):
        """Stores the other feature vectors in a separate variable (any leftover from `data` that are not in `cols`).

        Returns
        -------
        dataframe or ndarray
            "categorical" i.e. non-continuous feature vectors (as determined by `cols` attribute)
        """
        if type(self.data) == pd.DataFrame:
            return self.data.drop(self.cols, axis=1, inplace=False)
        elif type(self.data) == np.ndarray:
            ncols = list(range(self.data.shape[1]))
            cat_cols = [c for c in ncols if c not in self.cols]
            return self.data[:, cat_cols]

    def normalized_dataframe(self, normalized, join_data=True, rename=True):
        """Creates a new dataframe with the normalized data. Optionally combines with non-continuous vectors (original data) and appends `_scl` to the original column names for the ones that have been transformed.

        Parameters
        ----------
        normalized : dataframe
            normalized feature vectors
        join_data : bool, optional
            merge back with the original non-continuous data, by default True
        rename : bool, optional
            append '_scl' to normalized column names, by default True

        Returns
        -------
        dataframe
            dataframe of same shape as input data with continuous features normalized
        """
        try:
            idx = self.data.index
        except AttributeError:
            print("Cannot index a numpy array; use `normalized_matrix` instead.")
            return None
        if rename is True:
            newcols = [c + "_scl" for c in self.cols]
        else:
            newcols = self.cols
        data_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
        if join_data is True:
            data_norm = data_norm.join(self.categorical, how="left")
        return data_norm

    def normalized_matrix(self, normalized):
        """Concatenates arrays of normalized data with original non-continuous data along the y-axis (axis=1).

        Parameters
        ----------
        normalized : numpy.ndarray
            normalized data

        Returns
        -------
        numpy.ndarray
            array of same shape as input data, with continuous vectors normalized
        """
        if type(self.categorical) == pd.DataFrame:
            cat = self.categorical.values
        else:
            cat = self.categorical
        return np.concatenate((normalized, cat), axis=1)

    def normalizeX(self, normalized, join_data=True, rename=True):
        """Combines original non-continuous features/vectors with the transformed/normalized data. Determines datatype (array or dataframe) and calls the appropriate method.

        Parameters
        ----------
        normalized : dataframe or ndarray
            normalized data
        join_data : bool, optional
            merge back with non-continuous data, by default True
        rename : bool, optional
            append '_scl' to normalized column names, by default True

        Returns
        -------
        ndarray or dataframe
            array or dataframe of same shape and datatype as inputs, with continuous vectors/features normalized
        """
        if type(self.data) == pd.DataFrame:
            return self.normalized_dataframe(
                normalized, join_data=join_data, rename=rename
            )
        elif type(self.data) == np.ndarray:
            return self.normalized_matrix(normalized)
        else:
            return None


class PowerX(Transformer):
    """Applies Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to continuous feature vectors of a dataframe or numpy array. The `tx_data` attribute can be instantiated from a json file, dictionary or the input data itself. The training and test sets should be normalized separately (i.e. distinct class objects) to prevent data leakage when training a machine learning model. Loading the transform metadata from a json file allows you to transform a new input array (e.g. for predictions) without needing to access the original dataframe.

    Parameters
    ----------
    Transformer : class
        spacekit.preprocessor.transform.Transformer parent class

    Returns
    -------
    PowerX class object
        spacekit.preprocessor.transform.PowerX power transform subclass
    """

    def __init__(
        self,
        data,
        cols=[],
        tx_data=None,
        tx_file=None,
        save_tx=False,
        output_path=None,
        join_data=True,
        rename=True,
    ):
        super().__init__(
            data,
            cols=cols,
            tx_data=tx_data,
            tx_file=tx_file,
            save_tx=save_tx,
            output_path=output_path,
        )
        self.calculate_power()
        self.normalized = self.apply_power_matrix()
        self.Xt = super().normalizeX(
            self.normalized, join_data=join_data, rename=rename
        )

    def fitX(self):
        """Instantiates a scikit-learn PowerTransformer object and fits to the input data. If `tx_data` was passed as a kwarg or loaded from `tx_file`, the lambdas attribute for the transformer object will be updated to use these instead of calculated at the transform step.

        Returns
        -------
        PowerTransformer object
            transformer fit to the data
        """
        self.transformer = PowerTransformer(standardize=False).fit(self.continuous)
        self.transformer.lambdas_ = self.get_lambdas()
        return self.transformer

    def get_lambdas(self):
        """Instantiates the lambdas from file or dictionary if passed as kwargs; otherwise it uses the lambdas calculated in the transformX method. If transformX has not been called yet, returns None.

        Returns
        -------
        ndarray or float
            transform of multiple feature vectors returns an array of lambda values; otherwise a single vector returns a single (float) value.
        """
        if self.tx_data is not None:
            return self.tx_data["lambdas"]
        return self.transformer.lambdas_

    def transformX(self):
        """Applies a scikit-learn PowerTransform on the input data.

        Returns
        -------
        ndarray
            continuous feature vectors transformed via scikit-learn PowerTransform
        """
        return self.transformer.transform(self.continuous)

    def calculate_power(self):
        """Fits and transforms the continuous feature vectors using scikit learn PowerTransform. Calculates zero mean and unit variance for each vector as a separate step and stores these along with the lambdas in a dictionary `tx_data` attribute. This is so that the same normalization can be applied later for prediction inputs without requiring the original training data - otherwise it would be the same as using PowerTransform(standardize=True). Optionally, the calculated transform data can be stored in a json file on local disk.

        Returns
        -------
        self
            spacekit.preprocessor.transform.PowerX object with transformation metadata calculated for the input data and stored as attributes.
        """
        self.transformer = self.fitX()
        self.input_matrix = self.transformX()
        if self.tx_data is None:
            mu, sig = [], []
            for i in range(len(self.cols)):
                # normalized[:, i] = (v - m) / s
                mu.append(np.mean(self.input_matrix[:, i]))
                sig.append(np.std(self.input_matrix[:, i]))
            self.tx_data = {
                "lambdas": self.get_lambdas(),
                "mu": np.asarray(mu),
                "sigma": np.asarray(sig),
            }
            if self.save_tx is True:
                tx2 = {}
                for k, v in self.tx_data.items():
                    tx2[k] = list(v)
                _ = super().save_transformer_data(tx=tx2)
                del tx2
        return self

    def apply_power_matrix(self):
        """Transforms the input data. This method assumes we already have `tx_data` and a fit-transformed input_matrix (array of continuous feature vectors), which normally is done automatically when the class object is instantiated and `calculate_power` is called.

        Returns
        -------
        ndarray
            power transformed continuous feature vectors
        """
        nrows = self.continuous.shape[0]
        ncols = self.continuous.shape[1]
        self.normalized = np.empty((nrows, ncols))
        for i in range(ncols):
            v = self.input_matrix[:, i]
            m = self.tx_data["mu"][i]
            s = self.tx_data["sigma"][i]
            self.normalized[:, i] = (v - m) / s
        return self.normalized


def normalize_training_data(df, cols, X_train, X_test, X_val=None, output_path=None):
    """Apply Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to the training data, saving the transform metadata to json file on local disk and transforming the train, test and val sets separately (to prevent data leakage).

    Parameters
    ----------
    df : pandas dataframe
        training dataset
    cols: list
        column names or array index values of feature vectors to be transformed (i.e. continuous datatype features)
    X_train : ndarray
        training set feature inputs array
    X_test : ndarray
        test set feature inputs array
    X_val : ndarray, optional
        validation set inputs array, by default None

    Returns
    -------
    ndarrays
        normalized and scaled training, test, and validation sets
    """
    print("Applying Normalization (Leo-Johnson PowerTransform)")
    Px = PowerX(df, cols=cols, save_tx=True, output_path=output_path)
    X_train = PowerX(X_train, cols=cols, tx_data=Px.tx_data).Xt
    X_test = PowerX(X_test, cols=cols, tx_data=Px.tx_data).Xt
    if X_val is not None:
        X_val = PowerX(X_test, cols=cols, tx_data=Px.tx_data).Xt
        return X_train, X_test, X_val
    else:
        return X_train, X_test


# TODO: add/test single input array transformation (and reshape) to PowerX subclass, delete this one
class CalX(Transformer):
    def __init__(self, data, tx_file=None):
        super().__init__(data, cols=["n_files", "total_mb"], tx_file=tx_file)
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


# planned deprecation
def save_transformer_data(tx_data, output_path=None):
    """Save the transform metadata to a json file on local disk. Typical use-case is when you need to transform new inputs prior to generating a prediction but don't have access to the original dataset used to train the model.

    Parameters
    ----------
    tx_data : dictionary
        statistical metadata calculated when applying a transform to the training dataset; for PowerTransform this consists of lambdas, means and standard deviations for each continuous feature vector of the dataset.
    output_path : string, optional
        where to save the data as a json file on disk, by default None (defaults to current working directory)

    Returns
    -------
    string
        path where json file is saved on disk
    """
    if output_path is None:
        output_path = os.getcwd()
    else:
        os.makedirs(output_path, exist_ok=True)
    tx_file = f"{output_path}/tx_data"
    with open(tx_file, "w") as j:
        json.dump(tx_data, j)
    print("TX data saved as json file: ", tx_file)
    return tx_file


def split_sets(df, target="label", val=True):
    """Splits Pandas dataframe into feature (X) and target (y) train, test and validation sets.

    Parameters
    ----------
    df : Pandas dataframe
        preprocessed SVM regression test dataset
    target : str, optional
        target class label for alignment model predictions, by default "label"
    val : bool, optional
        create a validation set separate from train/test, by default True

    Returns
    -------
    Pandas dataframes
        features (X) and targets (y) split into train, test, and validation sets
    """
    print("Splitting Data ---> X-y ---> Train-Test-Val")
    y = df[target]
    X = df.drop(target, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )
    if val is True:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, shuffle=True, stratify=y_train
        )
        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        return X_train, X_test, y_train, y_test


# planned deprecation
def apply_power_transform(
    data,
    cols=["numexp", "rms_ra", "rms_dec", "nmatches", "point", "segment", "gaia"],
    output_path=None,
    save_tx=True,
):
    data_cont = data[cols]
    idx = data_cont.index
    tx = PowerTransformer(standardize=False)
    tx.fit(data_cont)
    input_matrix = tx.transform(data_cont)
    lambdas = tx.lambdas_
    normalized = np.empty((len(data), len(cols)))
    mu, sig = [], []
    for i in range(len(cols)):
        v = input_matrix[:, i]
        m, s = np.mean(v), np.std(v)
        x = (v - m) / s
        normalized[:, i] = x
        mu.append(m)
        sig.append(s)
    tx_data = {"lambdas": lambdas, "mu": np.asarray(mu), "sigma": np.asarray(sig)}
    if save_tx is True:
        tx2 = {}
        for k, v in tx_data.items():
            tx2[k] = list(v)
        _ = save_transformer_data(tx2, output_path=output_path)
        del tx2
    newcols = [c + "_scl" for c in cols]
    df_norm = pd.DataFrame(normalized, index=idx, columns=newcols)
    df = data.drop(cols, axis=1, inplace=False)
    df = df_norm.join(df, how="left")
    return df, tx_data


# planned deprecation
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


# for backward compatability with HSTCAL (planned deprecation)
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


# def normalize_training_data(df, X_train, X_test, X_val=None, output_path=None, save_tx=True):
#     """Apply Leo-Johnson PowerTransform (via scikit learn) normalization and scaling to the input features.

#     Parameters
#     ----------
#     df : pandas dataframe
#         training dataset
#     X_train : numpy array
#         training set feature inputs array
#     X_test : numpy array
#         test set feature inputs array
#     X_val : numpy array
#         validation set inputs array

#     Returns
#     -------
#     numpy arrays
#         normalized and scaled training, test, and validation sets
#     """
#     print("Applying Normalization (Leo-Johnson PowerTransform)")
#     _, px = apply_power_transform(df, output_path=output_path, save_tx=save_tx)
#     X_train = power_transform_matrix(X_train, px)
#     X_test = power_transform_matrix(X_test, px)
#     if X_val is not None:
#         X_val = power_transform_matrix(X_val, px)
#         return X_train, X_test, X_val
#     else:
#         return X_train, X_test


def normalize_training_images(X_tr, X_ts, X_vl=None):
    """Scale image inputs so that all pixel values are converted to a decimal between 0 and 1 (divide by 255).

    Parameters
    ----------
    X_tr : ndarray
        training set images
    test : ndarray
        test set images
    val : ndarray, optional
        validation set images, by default None

    Returns
    -------
    ndarrays
        image set arrays
    """
    X_tr /= 255.0
    X_ts /= 255.0
    if X_vl is not None:
        X_vl /= 255.0
        return X_tr, X_ts, X_vl
    else:
        return X_tr, X_ts


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
    """Adds an input corresponding to the running average over a set number of time steps. This helps the neural network to ignore high frequency noise by passing in a uniform 1-D filter and stacking the arrays.

    Parameters
    ----------
    matrix1 : numpy array
        e.g. X_train
    matrix2 : numpy array, optional
        e.g. X_test, by default None
    step_size : int, optional
        timesteps for 1D filter (e.g. 200), by default None
    axis : int, optional
        which axis to stack the arrays, by default 2

    Returns
    -------
    numpy array(s)
        2D array (original input array with a uniform 1d-filter as noise)
    """
    if step_size is None:
        step_size = 200

    # calc input for flux signal rolling avgs
    filter1 = uniform_filter1d(matrix1, axis=1, size=step_size)
    # store in array and stack on 2nd axis for each obs of X data
    matrix1 = np.stack([matrix1, filter1], axis=axis)

    if matrix2 is not None:
        filter2 = uniform_filter1d(matrix2, axis=1, size=step_size)
        matrix2 = np.stack([matrix2, filter2], axis=axis)
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
