import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
from keras.preprocessing import image
from tqdm import tqdm
import time
from spacekit.analyzer.track import stopwatch


class ArrayOps:
    def __init__(self, data_path=".", idx="ipst", target="mem_bin"):
        self.data_path = data_path
        self.idx = idx
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_idx = None

    """Pandas/Numpy File ops"""
    def load_train_test(self):
        self.X_train, self.y_train = np.load(f"{self.data_path}/X_train.npy"), np.load(
            f"{self.data_path}/y_train.npy"
        )
        self.X_test, self.y_test = np.load(f"{self.data_path}/X_test.npy"), np.load(
            f"{self.data_path}/y_test.npy"
        )
        if self.idx:
            self.test_idx = np.load(f"{self.data_path}/test_idx.npy", allow_pickle=True)
            if self.target:
                self.test_idx = pd.DataFrame(
                    np.argmax(self.y_test, axis=-1), index=self.idx, columns=[self.target]
                )
            return self.X_train, self.y_train, self.X_test, self.y_test, self.test_idx
        else:
            return self.X_train, self.y_train, self.X_test, self.y_test


    def save_train_test(self):
        np.save(f"{self.data_path}/X_train.npy", np.asarray(self.X_train))
        np.save(f"{self.data_path}/X_test.npy", np.asarray(self.X_test))
        np.save(f"{self.data_path}/y_train.npy", self. y_train)
        np.save(f"{self.data_path}/y_test.npy", self.y_test)
        np.save(f"{self.data_path}/test_idx.npy", np.asarray(self.test_idx.index))
        print("Train-test data saved as numpy arrays:\n")
        print(os.listdir(self.data_path))
    
    def save_compressed(self, arrs, names):
        for arr, name in list(zip(arrs, names)):
            np.savez_compressed(f"{self.data_path}/{name}.npz", arr)
        print("Train-test data saved as compressed numpy arrays:\n")
        print(os.listdir(self.data_path))
    
    def save_ensemble_data(self):
        X_train_mlp = np.asarray(self.X_train[0])
        X_train_img = np.asarray(self.X_train[1])
        X_test_mlp = np.asarray(self.X_test[0])
        X_test_img = np.asarray(self.X_test[1])
        y_train = np.asarray(self.y_train)
        y_test = np.asarray(self.y_test)
        test_idx = np.asarray(self.test_idx)
        arrays = [
            X_train_mlp, X_train_img, X_test_mlp, X_test_img, y_train, y_test, test_idx
            ]
        names = [
            "X_train_mlp", "X_train_img", "X_test_mlp", "X_test_img", "y_train", "y_test", "test_idx"
            ]
        self.save_compressed(arrays, names)

    def load_ensemble_data(self):
        X_train_mlp, X_train_img = np.load(f"{self.data_path}/X_train_mlp.npz"), np.load(
            f"{self.data_path}/X_train_img.npz"
        )
        X_test_mlp, X_test_img = np.load(f"{self.data_path}/X_test_mlp.npz"), np.load(
            f"{self.data_path}/X_test_img.npz"
        )
        self.X_train = [X_train_mlp["arr_0"], X_train_img["arr_0"]]
        self.X_test = [X_test_mlp["arr_0"], X_test_img["arr_0"]]
        self.y_train = np.load(f"{self.data_path}/y_train.npz")["arr_0"]
        self.y_test = np.load(f"{self.data_path}/y_test.npz")["arr_0"]
        self.test_idx = np.load(f"{self.data_path}/test_idx.npz")["arr_0"]


"""Image Ops"""

def unzip_images(zip_file):
    basedir = os.path.dirname(zip_file)
    key = os.path.basename(zip_file).split(".")[0]
    image_folder = os.path.join(basedir, key + "/")
    os.makedirs(image_folder, exist_ok=True)
    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(basedir)
    print(len(os.listdir(image_folder)))
    return image_folder

def read_channels(channels, w, h, d, exp=None, color_mode="rgb"):
    """Loads PNG image data and converts to 3D arrays.
    **args
    channels: tuple of image frames (original, source, gaia)
    w: width
    h: height
    d: depth
    **kwargs
    exp: "expand" dimensions: (exp, w, h, 3). Set to 3 for predictions, None for training (default)

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
    def __init__(self, img_path, w=128, h=128, d=9):
        self.img_path = img_path
        self.w = w
        self.h = h
        self.d = d

    def get_labeled_image_paths(self, i):
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

    def detector_training_images(self, X_data, exp=None):
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
