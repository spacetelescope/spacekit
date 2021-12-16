import pandas as pd
import numpy as np
import tensorflow as tf

# TODO: avoid using hardcoded settings (set for HST single visit mosaics)
SIZE = 128
DIM = 3
CH = 3
DEPTH = CH * DIM
SHAPE = (DIM, SIZE, SIZE, CH)

"""***REGRESSION TEST DATA AUGMENTATION FOR MLP***"""


# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


def laplacian_noise(x):
    return np.random.laplace(x)


def logistic_noise(x):
    return np.random.logistic(x)


def random_apply(func, x, p):
    r = tf.random.uniform([], minval=0, maxval=1)
    if r < p:
        return func(x)
    else:
        return x


def augment_random_noise(x):
    # augmentation transformations applied randomly to impose translational invariance.
    x = random_apply(laplacian_noise, x, p=0.8)
    x = random_apply(logistic_noise, x, p=0.8)
    return x


def augment_random_integer(x):
    n = np.random.randint(-1, 3)
    if x < 1:
        x = np.abs((x + n))
    else:
        x += n
    return x


def augment_data(xi):
    """Randomly apply noise to continuous data"""
    xi = np.array(
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
    return xi


def training_data_aug(X_train, X_test, X_val, y_train, y_test, y_val):
    xtr = np.empty(X_train.shape, dtype="float32")
    xts = np.empty(X_test.shape, dtype="float32")
    xvl = np.empty(X_val.shape, dtype="float32")
    X_train, X_test, X_val = X_train.values, X_test.values, X_val.values
    y_train, y_test, y_val = y_train.values, y_test.values, y_val.values
    for i in range(X_train.shape[0]):
        xtr[i] = augment_data(X_train[i])
    for i in range(X_test.shape[0]):
        xts[i] = augment_data(X_test[i])
    for i in range(X_val.shape[0]):
        xvl[i] = augment_data(X_val[i])
    X_train = np.concatenate([X_train, xtr, xts, xvl])
    y_train = np.concatenate([y_train, y_train, y_test, y_val])
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    return X_train, y_train


"""***IMAGE DATA PREP FOR 3DCNN***"""


def flip_horizontal(x):
    x = tf.image.flip_left_right(x)
    return x


def flip_vertical(x):
    x = tf.image.flip_up_down(x)
    return x


def rotate_k(x):
    # rotate 90 deg k times
    k = np.random.randint(3)
    x = tf.image.rot90(x, k)
    return x


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    # x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x  # .reshape(DIM, SIZE, SIZE, CH)


def augment_image(x, c=None):
    # the series of augmentation transformations
    # (except for random crops) need to be applied
    # randomly to impose translational invariance.
    from tensorflow.python.ops.numpy_ops import np_config

    np_config.enable_numpy_behavior()
    p = 0.5
    if x.shape[-1] == 9:
        x = x.reshape(3, SIZE, SIZE, 3)
    x = random_apply(flip_horizontal, x, p)
    x = random_apply(flip_vertical, x, p)
    x = random_apply(rotate_k, x, p)
    x = random_apply(color_jitter, x, p)
    if c == 9:
        return x.reshape(SIZE, SIZE, c)
    else:
        return x.reshape(DIM, SIZE, SIZE, CH)


def aug_generator(X_tr, X_ts, X_vl):
    c = 9
    xtr = np.empty(X_tr.shape, dtype="float32")
    xts = np.empty(X_ts.shape, dtype="float32")
    xvl = np.empty(X_vl.shape, dtype="float32")
    for i in range(X_tr.shape[0]):
        xtr[i] = augment_image(X_tr[i], c)
    for i in range(X_ts.shape[0]):
        xts[i] = augment_image(X_ts[i], c)
    for i in range(X_vl.shape[0]):
        xvl[i] = augment_image(X_vl[i], c)
    return xtr, xts, xvl


def expand_dims(Xtr, Xts, Xvl, d, w, h, c):
    ltr, lts, lvl = Xtr.shape[0], Xts.shape[0], Xvl.shape[0]
    Xtr = Xtr.reshape(ltr, d, w, h, c)
    Xts = Xts.reshape(lts, d, w, h, c)
    Xvl = Xvl.reshape(lvl, d, w, h, c)
    return Xtr, Xts, Xvl


def training_img_aug(train, test, val):
    X_tr, y_tr = train[1], train[2]
    X_ts, y_ts = test[1], test[2]
    X_vl, y_vl = val[1], val[2]
    xtr, xts, xvl = aug_generator(X_tr, X_ts, X_vl)
    X_tr = np.concatenate([X_tr, xtr, xts, xvl])
    y_tr = np.concatenate([y_tr, y_tr, y_ts, y_vl])
    X_tr, X_ts, X_vl = expand_dims(X_tr, X_ts, X_vl, DIM, SIZE, SIZE, CH)
    train_idx = pd.Index(np.concatenate([train[0], train[0], test[0], val[0]]))
    test_idx, val_idx = pd.Index(test[0]), pd.Index(val[0])
    train_Y = pd.Series(y_tr, index=train_idx)
    test_Y, val_Y = pd.Series(y_ts, index=test_idx), pd.Series(y_vl, index=val_idx)
    indX = (train_idx, test_idx, val_idx)
    indY = (train_Y, test_Y, val_Y)
    img_idx = [indX, indY]
    return img_idx, X_tr, y_tr, X_ts, y_ts, X_vl, y_vl
