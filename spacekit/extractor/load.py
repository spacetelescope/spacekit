import os
import pandas as pd
import numpy as np
import json
import pickle
import zipfile
from keras.preprocessing import image
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
from spacekit.analyzer.track import stopwatch


def load_datasets(filenames, index_col="index", column_order=None, verbose=1):
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
    if column_order:
        cols = [c for c in column_order if c in df.columns]
        df = df[cols]
    print("Input Shape: ", df.shape)
    if verbose:
        print(df.columns)
    return df


def stratified_splits(df, target="label", v=0.85):
    """Splits Pandas dataframe into feature (X) and target (y) train, test and validation sets.

    Parameters
    ----------
    df : Pandas dataframe
        preprocessed SVM regression test dataset
    target : str, optional
        target class label for alignment model predictions, by default "label"
    test_size : int, optional
        size of the test set, by default 0.2
    val_size : int, optional
        create a validation set separate from train/test, by default 0.1

    Returns
    -------
    tuples of Pandas dataframes
        data, labels: features (X) and targets (y) split into train, test, validation sets
    """
    print("Splitting Data ---> X-y ---> Train-Test-Val")
    seed = np.random.randint(1, 42)
    y = df[target]
    X = df.drop(target, axis=1, inplace=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    X_val, y_val = np.asarray([]), np.asarray([])
    if v > 0:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=1 - v,
                shuffle=True,
                stratify=y_train,
                random_state=seed,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                shuffle=True,
                stratify=y_train,
                random_state=seed,
            )
    data = (X_train, X_test, X_val)
    labels = (y_train, y_test, y_val)
    return data, labels


def load_npz(npz_file="data/img_data.npz"):
    try:
        img_data = np.load(npz_file)
        X = img_data["images"]
        y = img_data["labels"]
        index = img_data["index"]
        img_data.close()
        return (index, X, y)
    except Exception as e:
        print(e)
        return None


def save_npz(i, X, y, npz_file="data/img_data.npz"):
    """Store compressed data to disk"""
    np.savez(npz_file, index=i, images=X, labels=y)


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


class ImageIO:
    """Parent Class for image file input/output operations"""

    def __init__(self, img_path, format="png", data=None):
        self.img_path = img_path
        self.format = self.check_format(format)
        self.data = data

    def check_format(self, format):
        """Checks the format type of ``img_path`` (``png``, ``jpg`` or ``npz``) and initializes the ``format`` attribute accordingly.

        Parameters
        ----------
        format : str
            (``png``, ``jpg`` or ``npz``)

        Returns
        -------
        str
            (``png``, ``jpg`` or ``npz``)
        """
        sfx = self.img_path.split(".")[-1]
        if sfx == "npz":
            return "npz"
        else:
            return format

    def load_npz(self, npz_file=None, keys=["index", "images", "labels"]):
        if npz_file is None:
            npz_file = self.img_path
            try:
                img_data = np.load(npz_file)
                index = img_data[keys[0]]
                X = img_data[keys[1]]
                if len(keys) > 2:
                    y = img_data[keys[2]]
                    img_data.close()
                    return (index, X, y)
                else:
                    img_data.close()
                    return index, X
            except Exception as e:
                print(e)
                return None

    def load_multi_npz(self, i="img_index.npz", X="img_data.npz", y="img_labels.npz"):
        """Load numpy arrays from individual feature/image data, label and index compressed files on disk"""
        (X_train, X_test, X_val) = self.load_npz(
            npz_file=X, keys=["X_train", "X_test", "X_val"]
        )
        (y_train, y_test, y_val) = self.load_npz(
            npz_file=y, keys=["y_train", "y_test", "y_val"]
        )
        (train_idx, test_idx, val_idx) = self.load_npz(
            npz_file=i, keys=["train_idx", "test_idx", "val_idx"]
        )
        train = (train_idx, X_train, y_train)
        test = (test_idx, X_test, y_test)
        val = (val_idx, X_val, y_val)
        return train, test, val

    def save_npz(self, i, X, y, npz_file="data/img_data.npz"):
        """Store compressed data to disk"""
        np.savez(npz_file, index=i, images=X, labels=y)

    def save_multi_npz(self, train, test, val, data_path="data"):
        np.savez(
            f"{data_path}/index.npz",
            train_idx=train[0],
            test_idx=test[0],
            val_idx=val[0],
        )
        np.savez(
            f"{data_path}/images.npz", X_train=train[1], X_test=test[1], X_val=val[1]
        )
        np.savez(
            f"{data_path}/labels.npz", y_train=train[2], y_test=test[2], y_val=val[2]
        )

    def split_arrays(self, data, t=0.6, v=0.85):
        if type(data) == pd.DataFrame:
            sample = data.sample(frac=1)
        else:
            sample = data
        if v > 0:
            return np.split(sample, [int(t * len(data)), int(v * len(data))])
        else:
            arrs = np.split(sample, [int(t * len(data))])
            arrs.append(np.asarray([]))
            return arrs

    def split_arrays_from_npz(self, v=0.85):
        """Loads images (X), labels (y) and index (i) from a single .npz compressed numpy file. Splits into train, test, val sets using 70-20-10 ratios.

        Returns
        -------
        tuples
            train, test, val tuples of numpy arrays. Each tuple consists of an index, feature data (X, for images these are the actual pixel values) and labels (y).
        """
        (index, X, y) = self.load_npz()
        train_idx, test_idx, val_idx = self.split_arrays(index, v=v)
        X_train, X_test, X_val = self.split_arrays(X, v=v)
        y_train, y_test, y_val = self.split_arrays(y, v=v)
        train = (train_idx, X_train, y_train)
        test = (test_idx, X_test, y_test)
        val = (val_idx, X_val, y_val)
        return train, test, val

    def split_df_from_arrays(self, train, test, val, target="label"):
        if self.data is None:
            return
        X_train = self.data.loc[train[0]].drop(target, axis=1, inplace=False)
        X_test = self.data.loc[test[0]].drop(target, axis=1, inplace=False)
        y_train = self.data.loc[train[0]][target]
        y_test = self.data.loc[test[0]][target]
        X_val, y_val = pd.DataFrame(), pd.DataFrame()
        if len(val[0]) > 0:
            X_val = self.data.loc[val[0]].drop(target, axis=1, inplace=False)
            y_val = self.data.loc[val[0]][target]
        X = (X_train, X_test, X_val)
        y = (y_train, y_test, y_val)
        return X, y


class SVMImageIO(ImageIO):
    """Subclass for loading Single Visit Mosaic total detection .png images from local disk into numpy arrays and performing initial preprocessing and labeling for training a CNN or generating predictions on unlabeled data.

    Parameters
    ----------
    ImageIO: class
        ImageIO parent class
    """

    def __init__(
        self,
        img_path,
        w=128,
        h=128,
        d=9,
        inference=True,
        format="png",
        data=None,
        target="label",
        v=0.85,
    ):
        """Instantiates an SVMFileIO object.

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
        inference: bool, optional
            determines how to load images (set to False for training), by default True
        format: str, optional
            format type of image file(s), ``png``, ``jpg`` or ``npz``, by default "png"
        data: dataframe, optional
            used to load mlp data inputs and split into train/test/validation sets, by default None
        target: str, optional
            name of the target column in dataframe, by default "label"
        v: float, optional
            size ratio for validation set, by default 0.85
        """
        super().__init__(img_path, format=format, data=data)
        self.w = w
        self.h = h
        self.d = d
        self.inference = inference
        self.target = target
        self.v = v

    def load(self):
        if self.inference is True:  # idx, images
            if self.format in ["png", "jpg"]:
                return self.detector_prediction_images(self.data, exp=3)
            elif self.format == "npz":
                return super().load_npz(keys=["index", "images"])
        else:
            if self.format in ["png", "jpg"]:
                X, y = stratified_splits(self.data, target=self.target, v=self.v)
                train, test, val = self.load_from_data_splits(*X)
            elif self.format == "npz":
                train, test, val = super().split_arrays_from_npz(v=self.v)
                X, y = super().split_df_from_arrays(
                    train, test, val, target=self.target
                )
            return (X, y), (train, test, val)

    def load_from_data_splits(self, X_train, X_test, X_val):
        """Read in train/test files and produce X-y data splits.

        Parameters
        ----------
        X_train : numpy.ndarray
            training image inputs
        X_test : numpy.ndarray
            test image inputs
        X_val : numpy.ndarray
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
        if len(X_val) > 0:
            print("\n*** Validation Set ***")
            val = self.detector_training_images(X_val)
        else:
            val = [X_val, X_val, X_val]
        end = time.time()
        print("\n")
        stopwatch("LOADING IMAGES", t0=start, t1=end)
        print("\n[i] Length of Splits:")
        print(f"X_train={len(train[1])}, X_test={len(test[1])}, X_val={len(val[1])}")
        return train, test, val

    def get_labeled_image_paths(self, i):
        """Creates lists of negative and positive image filepaths, assuming the image files are in subdirectories named according to the class labels e.g. "0" and "1" (Similar to how Keras ``flow_from_directory`` works). Note: this method expects 3 images in the subdirectory, two of which have suffices _source and _gaia appended, and a very specific path format: ``{img_path}/{label}/{i}/{i}_{suffix}.png`` where ``i`` is typically the full name of the visit. This may be made more flexible in future versions but for now is more or less hardcoded for SVM images generated by ``spacekit.skopes.hst.svm.prep`` or ``corrupt`` modules.

        Parameters
        ----------
        i : str
            image filename

        Returns
        -------
        tuples
            image filenames for each image type (original, source, gaia)
        """
        fmt = self.format
        neg = (
            f"{self.img_path}/0/{i}/{i}.{fmt}",
            f"{self.img_path}/0/{i}/{i}_source.{fmt}",
            f"{self.img_path}/0/{i}/{i}_gaia.{fmt}",
        )
        pos = (
            f"{self.img_path}/1/{i}/{i}.{fmt}",
            f"{self.img_path}/1/{i}/{i}_source.{fmt}",
            f"{self.img_path}/1/{i}/{i}_gaia.{fmt}",
        )
        return neg, pos

    def detector_training_images(self, X_data, exp=None):
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
            img.append(read_channels([ch1, ch2, ch3], self.w, self.h, self.d, exp=exp))
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
        fmt = self.format
        image_files = []
        idx = list(X_data.index)
        for i in idx:
            img_frames = (
                f"{self.img_path}/{i}/{i}.{fmt}",
                f"{self.img_path}/{i}/{i}_source.{fmt}",
                f"{self.img_path}/{i}/{i}_gaia.{fmt}",
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
        X_img = np.array(img, np.float32)
        end = time.time()
        stopwatch("LOADING IMAGES", t0=start, t1=end)
        print("Inputs: ", X_img.shape[0])
        print("Dimensions: ", X_img.shape[1])
        print("Width: ", X_img.shape[2])
        print("Height: ", X_img.shape[3])
        print("Channels: ", X_img.shape[4])
        print("Input Shape: ", X_img.shape)
        return idx, X_img


def save_dct_to_txt(data_dict):
    """Saves the key-value pairs of a dictionary to text files on local disk, with each key as a filename and its value(s) as the contents of that file.

    Parameters
    ----------
    data_dict : dict
        dictionary containing keys as filenames and values as the contents to be saved to a text file.

    Returns
    -------
    list
        list of paths to each file saved to local disk.
    """
    keys = []
    for filename, data in data_dict.items():
        key = f"{filename}.txt"
        keys.append(key)
        with open(f"{key}", "w") as f:
            for item in data:
                f.writelines(f"{item}\n")
    print(f"Saved file keys:\n {keys}")
    return keys


def save_dict(data_dict, df_key=None):
    keys = []
    for key, data in data_dict.items():
        filename = f"{key}.txt"
        with open(filename, "w") as f:
            try:
                json.dump(data, f)
            except Exception as e:
                print(e)
                f.writelines(data)
        keys.append(filename)
    if df_key is not None:
        keys.append(df_key)
    print(f"File keys:\n {keys}")
    return keys


def save_json(data, name):
    with open(name, "w") as fp:
        json.dump(data, fp)
    print(f"\nJSON file saved:\n {os.path.abspath(name)}")


def save_dataframe(df, df_key, index_col="ipst"):
    df[index_col] = df.index
    df.to_csv(df_key, index=False)
    print(f"Dataframe saved as: {df_key}")
    df.set_index(index_col, drop=True, inplace=True)
    return df


def save_to_pickle(data_dict, target_col=None, df_key=None):
    keys = []
    for k, v in data_dict.items():
        if target_col is not None:
            os.makedirs(f"{target_col}", exist_ok=True)
            key = f"{target_col}/{k}"
        else:
            key = k
        with open(key, "wb") as file_pi:
            pickle.dump(v, file_pi)
            print(f"{k} saved as {key}")
            keys.append(key)
    if df_key is not None:
        keys.append(df_key)
    print(f"File keys:\n {keys}")
    return keys


def zip_subdirs(top_path, zipname="models.zip"):
    file_paths = []
    for root, _, files in os.walk(top_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    print("Zipping model files:")
    with zipfile.ZipFile(zipname, "w") as zip_ref:
        for file in file_paths:
            zip_ref.write(file)
            print(file)
