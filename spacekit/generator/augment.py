import pandas as pd
import numpy as np
import tensorflow as tf


# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


"""***REGRESSION TEST DATA AUGMENTATION FOR MLP***"""


def laplacian_noise(x):
    """Calculates laplacian value relative to the input.

    Parameters
    ----------
    x : float
        input value

    Returns
    -------
    float
        random laplacian value relative to input.
    """
    return np.random.laplace(x)


def logistic_noise(x):
    """Calculates random logistic value relative to the input.

    Parameters
    ----------
    x : float
        input value

    Returns
    -------
    float
        random logistic value relative to input.
    """
    return np.random.logistic(x)


def random_apply(func, x, p):
    """Determine whether or not to apply a transformation according to a randomly calculated probability value.

    Parameters
    ----------
    func : function
        augmentation function to apply if random number is less than p.
    x : float
        input value
    p : float
        probability between 0 and 1 used to determine whether or not to apply the augmentation.

    Returns
    -------
    float
        transformed input (or original input if no transformation is performed).
    """
    r = tf.random.uniform([], minval=0, maxval=1)
    if r < p:
        return func(x)
    else:
        return x


def augment_random_noise(x):
    """Random application of laplacian and/or logistic noise calculated relative to the input value. Augmentation transformations are applied randomly to impose translational invariance.

    Parameters
    ----------
    x : float
        input value

    Returns
    -------
    float
        transformed input value
    """
    x = random_apply(laplacian_noise, x, p=0.8)
    x = random_apply(logistic_noise, x, p=0.8)
    return x


def augment_random_integer(x):
    """Randomly apply augmentations to integer values. Assumes final values must be absolute/non-negative.

    Parameters
    ----------
    x : int
        input value (integer)

    Returns
    -------
    int
        absolute (non-negative) augmented integer
    """
    n = np.random.randint(-1, 3)
    if x < 1:
        x = np.abs((x + n))
    else:
        x += n
    return x


def augment_data(xi):
    """Randomly apply noise to input data. For now this function is designed for SVM ensemble MLP input data. For other uses, the function tries to determine which augmentation operation to apply according to the data type - however it may not make sense for all the inputs to be augmented (e.g. if they are binary/boolean) so this should be used with caution (future updates this will be generalized properly for other use cases).

    Parameters
    ----------
    xi : numpy array
        1d-array of input data

    Returns
    -------
    numpy array
        input data with random augmentations applied
    """
    if len(xi) == 10:
        return np.array(
            [
                augment_random_integer(xi[0]),
                augment_random_noise(xi[1]),
                augment_random_noise(xi[2]),
                augment_random_integer(xi[3]),
                augment_random_noise(xi[4]),
                augment_random_noise(xi[5]),
                augment_random_integer(xi[6]),
                xi[7],
                xi[8],
                xi[9],
            ]
        )
    else:
        x2 = []
        for x in xi:
            if type(x) == int:
                x2.append(augment_random_integer(x))
            elif type(x) == float:
                x2.append(augment_random_noise[x])
        return np.asarray(x2)


def training_data_aug(X_train, y_train):
    """Perform data augmentation on the training set

    Parameters
    ----------
    X_train : dataframe
        training set features
    y_train : dataframe
        training set target labels

    Returns
    -------
    dataframes
        augmented training data and target labels combined with original set (2x observations)
    """
    xtr = np.empty(X_train.shape, dtype="float32")
    X_train = X_train.values
    y_train = y_train.values
    for i in range(X_train.shape[0]):
        xtr[i] = augment_data(X_train[i])
    X_train = np.concatenate([X_train, xtr])
    y_train = np.concatenate([y_train, y_train])
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    return X_train, y_train


"""***IMAGE AUGMENTATION***"""


def flip_horizontal(x):
    """Horizontal (x-axis) image flip

    Parameters
    ----------
    x : numpy array
        input image array

    Returns
    -------
    numpy array
        horizontally flipped image
    """
    x = tf.image.flip_left_right(x)
    return x


def flip_vertical(x):
    """Vertical (y-axis) image flip

    Parameters
    ----------
    x : numpy array
        input image array

    Returns
    -------
    numpy array
        vertically flipped image
    """
    x = tf.image.flip_up_down(x)
    return x


def rotate_k(x):
    """Rotate image array k times by 90 degrees. K is a number between 0 and 3 determined randomly when the function is called.

    Parameters
    ----------
    x : numpy array
        input image array

    Returns
    -------
    numpy array
        randomly rotated image
    """
    k = np.random.randint(3)
    x = tf.image.rot90(x, k)
    return x


def color_jitter(x, brightness=0.4, contrast=0.4, saturation=0.4, hue=None):
    """Randomly applies a combination of alterations to the image brightness, contrast, saturation and hue. Note: a clipping function is performed at the end since affine transformations can disturb the natural range of RGB images.

    Parameters
    ----------
    x : numpy array
        input image array
    brightness : float, optional
        decimal percentage strength applied to brightness level, by default 0.4
    contrast: float, optional
        decimal percentage strength applied to contrast level, by default 0.4
    saturation: float, optional
        decimal percentage strength applied to saturation level, by default 0.4
    hue: float, optional
        decimal percentage strength applied to hue (e.g. 0.1), by default None

    Returns
    -------
    numpy array
        transformed image input array
    """
    x = tf.image.random_brightness(x, max_delta=0.8 * brightness)
    x = tf.image.random_contrast(x, lower=1 - 0.8 * contrast, upper=1 + 0.8 * contrast)
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * saturation, upper=1 + 0.8 * saturation
    )
    if hue is not None:
        x = tf.image.random_hue(x, max_delta=0.2 * hue)
    x = tf.clip_by_value(x, 0, 255)
    return x


def augment_image(x, c=None, w=128, h=128, ch=3, dim=3):
    """Randomly applies a series of augmentation transformations. Except for random crops, random application is required
    to impose translational invariance.

    Parameters
    ----------
    x : numpy array
        input image array
    c : int, optional
        depth (product of dim and ch), by default None
    w : int, optional
        image width, by default 128
    h : int, optional
        image height, by default 128
    ch : int, optional
        channels, by default 3
    dim : int, optional
        dimensions (or volume of image frames per input), by default 3

    Returns
    -------
    numpy array
        random augmentation of input image
    """
    from tensorflow.python.ops.numpy_ops import np_config

    np_config.enable_numpy_behavior()
    depth = ch * dim
    p = 0.5
    if x.shape[-1] == depth:
        x = x.reshape(dim, w, h, ch)
    x = random_apply(flip_horizontal, x, p)
    x = random_apply(flip_vertical, x, p)
    x = random_apply(rotate_k, x, p)
    x = random_apply(color_jitter, x, p)
    if c == depth:
        return x.reshape(w, h, c)
    else:
        return x.reshape(dim, w, h, ch)


def expand_dims(X, dim=3, w=128, h=128, ch=3):
    """Expand flattened array back into its constituent dimensions.

    Parameters
    ----------
    X : numpy array
        array of image arrays
    dim : int, optional
        dimensions (volume of image frames per input), by default 3
    w : int, optional
        single image width, by default 128
    h : int, optional
        single image height, by default 128
    ch : int, optional
        channels, by default 3

    Returns
    -------
    numpy array of shape: ``(X.shape[0], dim, w, h, ch)``
        Reshaped multi-dimensional image array
    """
    return X.reshape(X.shape[0], dim, w, h, ch)


def aug_generator(X, c=9, combine=False):
    """Generates randomly performed image augmentations. Optionally concatenates this data with original.

    Parameters
    ----------
    X : numpy array
        array of image arrays(pixel vlues)
    c : int, optional
        channel depth (product of dim and ch), by default 9
    combine : bool, optional
        concatenate original with augmented data, by default False

    Returns
    -------
    numpy array
        randomly augmented image data (merged with original data if combine set to True).
    """
    Xa = np.empty(X.shape, dtype="float32")
    for i in range(X.shape[0]):
        Xa[i] = augment_image(X[i], c)
    if combine is True:
        Xa = np.concatenate([X, Xa])
    return Xa


def image_index_labels(index, y, aug=False):
    """Creates Pandas Index and Series from numpy arrays (used for calculating FNFP data in ``spacekit.analyzer.compute``)

    Parameters
    ----------
    index : ndarray
        index of image names
    y : ndarray
        target class values
    aug : bool, optional
        concatenate (double) the index for training data, by default False

    Returns
    -------
    tuple of pd.Index and pd.Series
        inputs converted into index and series
    """
    if aug is False:
        idx = pd.Index(index)
        y_series = pd.Series(y, index=idx)
        return (idx, y_series)
    else:  # double y_train to match augmented X
        idx = pd.Index(np.concatenate([index, index]))
        y_aug = np.concatenate([y, y])
        y_labels = pd.Series(y_aug, index=idx)
        return (idx, y_labels), y_aug


def nested_image_index(tr, ts, vl=()):
    """creates a list of nested tuples for training, test, and validation Indices and Index-label Series

    Parameters
    ----------
    tr : tuple
        (training index, training index-label series)
    ts : tuple
        (test index, test index-label series)
    vl : tuple, optional
        (val index, val index-label series), by default ()

    Returns
    -------
    list
        list of nested tuples [(indices), (tuples)]
    """
    if len(vl) > 0:
        indices = (tr[0], ts[0], vl[0])
        labels = (tr[1], ts[1], vl[1])
    else:
        indices = (tr[0], ts[0])
        labels = (tr[1], ts[1])
    return [indices, labels]


def training_img_aug(train, test, val=None, dim=3, w=128, h=128, ch=3):
    """Perform image data augmentation on the training set

    Parameters
    ----------
    train : tuple
        training set tuple of (train_index, X_train, y_train)
    test : tuple
        test set tuple of (test_index, X_test, y_test)
    val : tuple, optional
        val set tuple of (val_index, X_val, y_val), by default None
    dim : int, optional
        dimensions (volume or number of image frames per observation), by default 3
    w : int, optional
        single image width, by default 128
    h : int, optional
        single image height, by default 128
    ch : int, optional
        channels (rgb: 3, grayscale: 1), by default 3

    Returns
    -------
    list of tuples of Pandas Index, numpy arrays
        combined image index of train, test, val; combined original and augmented training data, reshaped test and val data. If no validation data is passed in the ``val`` arg, then only train and test are returned.
    """
    depth = ch * dim
    X_tr = aug_generator(train[1], c=depth, combine=True)
    X_tr = expand_dims(X_tr, dim=dim, w=w, h=h, ch=ch)
    y_train_idx, y_tr = image_index_labels(train[0], train[2], aug=True)
    train = (X_tr, y_tr)

    X_ts = expand_dims(test[1], dim=dim, w=w, h=h, ch=ch)
    y_test_idx = image_index_labels(test[0], test[2])
    test = (X_ts, test[2])

    if val is not None and len(val[0]) > 0:
        X_vl = expand_dims(val[1], dim=dim, w=w, h=h, ch=ch)
        y_val_idx = image_index_labels(val[0], val[2])
        val = (X_vl, val[2])
    else:
        y_val_idx = ()

    img_idx = nested_image_index(y_train_idx, y_test_idx, vl=y_val_idx)
    return img_idx, train, test, val
