import pandas as pd
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf


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


def make_tensors(X_train, y_train, X_test, y_test):
    """Convert Arrays to Tensors"""
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
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
