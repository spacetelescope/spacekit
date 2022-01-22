import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import time
from spacekit.analyzer.track import stopwatch


def load_datasets(filenames, index_col="index"):
    """Import one or more dataframes from csv files and merge along the 0 axis (rows / horizontal). Assumes the datasets use the same index_col name and identical column names (although this is not strictly required) since this function does not handle missing data or NaNs.

    Parameters
    ----------
    filenames : list
        path(s) to csv files of saved dataframes.
    index_col : str, optional
        name of the index column to set

    Returns
    -------
    DataFrame
        Labeled dataframe loaded from csv file(s).
    """
    if len(filenames) == 1:
        df = pd.read_csv(filenames[0], index_col=index_col)
    else:
        dfs = []
        for filename in filenames:
            dfs.append(pd.read_csv(filename, index_col=index_col))
        df = pd.concat([d for d in dfs], axis=0)
    return df

def load_npz(npz_file="data/img_data.npz"):
    try:
        img_data = np.load(npz_file)
        X = img_data['images']
        y = img_data['labels']
        index = img_data["index"]
        img_data.close()
        return (index, X, y)
    except Exception as e:
        print(e)
        return None


def load_multi_npz(train="data/img_data.npz", test="data/img_labels.npz", val="data/img_index.npz"):
    """Load compressed numpy data from disk"""
    train_data, test_data, val_data = None, None, None
    if train:
        train_data = load_npz(npz_file=train)
    if test:
        test_data = load_npz(npz_file=test)
    if val:
        val_data = load_npz(npz_file=val)
    return train_data, test_data, val_data


def save_npz(X, y, i, npz_file="data/img_data.npz"):
    """Store compressed data to disk"""
    np.savez(npz_file, images=X, labels=y, index=i)



def load_compressed(self):
    """Store compressed data to disk"""
    X_data = np.load(f"{self.data_path}/X.npz")

    self.X_train = X_data["X_train"]
    self.X_test = X_data["X_test"]
    X_data.close()

    y_data = np.load(f"{self.data_path}/y.npz")
    self.y_train = y_data["y_train"]
    self.y_test = y_data["y_test"]
    y_data.close()

    idx = np.load(f"{self.data_path}/idx.npz")
    self.test_idx = idx["test_idx"]
    return self


# TODO
# def save_compressed(self, y_dict=None, idx_dict=None):
#     """Store compressed data to disk"""
#     np.savez(f"{self.data_path}/X.npz", X_train=self.X_train, X_test=self.X_test)
#     if y_dict:
#         np.savez(f"{self.data_path}/y.npz", **y_dict)
#     else:
#         np.savez(
#             f"{self.data_path}/y.npz", y_train=self.y_train, y_test=self.y_test
#         )
#     if idx_dict:
#         np.savez(f"{self.data_path}/idx.npz", **idx_dict)
#     else:
#         np.savez(f"{self.data_path}/idx.npz", test_idx=self.test_idx)


"""Image Ops"""


def read_channels(channels, w, h, d, exp=None, color_mode="rgb"):
    """Loads PNG image data and converts to 3D arrays.

    Parameters
    ----------
    channels : tuple
        image frames (original, source, gaia)
    w : int
        image width
    h : int
        image height
    d : int
        depth (number of image frames)
    exp : int, optional
        expand array dimensions ie reshape to (exp, w, h, 3), by default None
    color_mode : str, optional
        RGB (3 channel images) or grayscale (1 channel), by default "rgb". SVM predictions requires exp=3; set to None for training.

    Returns
    -------
    numpy array
        image pixel values as array
    """
    t = (w, h)
    image_frames = [
        image.load_img(c, color_mode=color_mode, target_size=t) for c in channels
    ]
    img = np.array([image.img_to_array(i) for i in image_frames])
    if exp is None:
        img = img.reshape(w, h, d)
    else:
        img = img.reshape(exp, w, h, 3)
    return img


class SVMImages:
    """Class for loading Single Visit Mosaic total detection .png images from local disk into numpy arrays and performing initial preprocessing and labeling for training a CNN or generating predictions on unlabeled data."""

    def __init__(self, img_path, w=128, h=128, d=9, exp=None):
        """Instantiates an SVMImages class object.

        Parameters
        ----------
        img_path : string
            path to local directory containing png files
        w : int, optional
            image pixel width, by default 128
        h : int, optional
            image pixel height, by default 128
        d : int, optional
            channel depth, by default 9
        """
        self.img_path = img_path
        self.w = w
        self.h = h
        self.d = d
        self.exp = exp

    def get_labeled_image_paths(self, i):
        """Creates lists of negative and positive image filepaths, assuming the image files are in subdirectories named according to the class labels (e.g. "0" and "1").

        Parameters
        ----------
        i : str
            image filename

        Returns
        -------
        tuples
            image filenames for each image type (original, source, gaia)
        """
        neg = (
            f"{self.img_path}/0/{i}/{i}.png",
            f"{self.img_path}/0/{i}/{i}_source.png",
            f"{self.img_path}/0/{i}/{i}_gaia.png",
        )
        pos = (
            f"{self.img_path}/1/{i}/{i}.png",
            f"{self.img_path}/1/{i}/{i}_source.png",
            f"{self.img_path}/1/{i}/{i}_gaia.png",
        )
        return neg, pos

    def detector_training_images(self, X_data):
        """Load image files from class-labeled folders containing pngs into numpy arrays. Image arrays are **not** reshaped since this assumes data augmentation will be performed at training time.

        Parameters
        ----------
        X_data : Pandas dataframe
            input data (assumes index values are the image filenames)
        exp : int, optional
            expand image array shape into its constituent frame dimensions, by default None

        Returns
        -------
        tuple
            index, image input array, image class labels: (idx, X, y)
        """
        idx = list(X_data.index)
        files, labels = [], []
        for i in idx:
            neg, pos = self.get_labeled_image_paths(i)
            if os.path.exists(neg[0]):
                files.append(neg)
                labels.append(0)
            elif os.path.exists(pos[0]):
                files.append(pos)
                labels.append(1)
            else:
                # print(f"missing: {i}")
                idx.remove(i)
        img = []
        for ch1, ch2, ch3 in tqdm(files):
            img.append(read_channels([ch1, ch2, ch3], self.w, self.h, self.d, exp=self.exp))
        X, y = np.array(img, np.float32), np.array(labels)
        return (idx, X, y)

    def detector_prediction_images(self, X_data, exp=3):
        """Load image files from pngs into numpy arrays. Image arrays are reshaped into the appropriate dimensions for generating predictions in a pre-trained image CNN (no data augmentation is performed).

        Parameters
        ----------
        X_data : Pandas dataframe
            input data (assumes index values are the image filenames)
        exp : int, optional
            expand image array shape into its constituent frame dimensions, by default 3

        Returns
        -------
        Pandas Index, numpy array
            image name index, arrays of image pixel values
        """
        image_files = []
        idx = list(X_data.index)
        for i in idx:
            img_frames = (
                f"{self.img_path}/{i}/{i}.png",
                f"{self.img_path}/{i}/{i}_source.png",
                f"{self.img_path}/{i}/{i}_gaia.png",
            )
            if os.path.exists(img_frames[0]):
                image_files.append(img_frames)
            else:
                idx.remove(i)
        start = time.time()
        stopwatch("LOADING IMAGES", t0=start)
        img = []
        for ch1, ch2, ch3 in tqdm(image_files):
            img.append(read_channels([ch1, ch2, ch3], self.w, self.h, self.d, exp=exp))
        images = np.array(img, np.float32)
        end = time.time()
        stopwatch("LOADING IMAGES", t0=start, t1=end)
        return idx, images
    
    def split_from_npz(self):
        """Loads images, labels and index from a single compressed py file (.npz). Splits into train, test, val sets using ratios of 70-20-10 percent.

        Returns
        -------
        nested lists
            train, test, val nested lists each containing an index of the visit names and png image data as numpy arrays.
        """
        (index, X, y) = load_npz(npz_file=self.img_path)
        size_train = int(np.floor(X.shape[0] * 0.7))
        size_test = int(np.floor(X.shape[0] * 0.2))
        size_val = size_train + size_test
        train = (index[:size_train], X[:size_train], y[:size_train])
        test = (index[size_train:size_test], X[size_train:size_test], y[size_train:size_test])
        val = (index[size_val:], X[size_val:], y[size_val:])
        return train, test, val
    
    def load_split_training(self, X_train, X_test, X_val):
        """Read in train/test files and produce X-y data splits.

        Parameters
        ----------
        X_train : numpy array
            training image inputs
        X_test : [type]
            test image inputs
        X_val : [type]
            validation image inputs

        Returns
        -------
        nested lists
            train, test, val nested lists each containing an index of the visit names and png image data as numpy arrays.
        """
        start = time.time()
        stopwatch("LOADING IMAGES", t0=start)
        print("\n*** Training Set ***")
        train = self.detector_training_images(X_train)
        print("\n*** Test Set ***")
        test = self.detector_training_images(X_test)
        print("\n*** Validation Set ***")
        val = self.detector_training_images(X_val)
        end = time.time()
        print("\n")
        stopwatch("LOADING IMAGES", t0=start, t1=end)
        print("\n[i] Length of Splits:")
        print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")
        return train, test, val
