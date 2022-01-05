import os
import pandas as pd
import numpy as np

# from zipfile import ZipFile
from keras.preprocessing import image
from tqdm import tqdm
import time
from spacekit.analyzer.track import stopwatch


class ArrayOps:
    def __init__(self, data_path=".", idx=None, targets=None):
        self.data_path = data_path
        self.idx = idx
        self.targets = targets
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_idx = None

    """Pandas/Numpy File ops"""

    def save_train_test(self, target=None):
        self.save_X_train_test()
        if self.test_idx:
            self.save_test_index(target=target)
        self.save_y_train_test(target=target)
        print("Train-test data saved as numpy arrays:\n")
        print(os.listdir(self.data_path))

    # def save_train_test(self, target=None):
    #     np.save(f"{self.data_path}/X_train.npy", np.asarray(self.X_train))
    #     np.save(f"{self.data_path}/X_test.npy", np.asarray(self.X_test))
    #     if self.test_idx:
    #         np.save(f"{self.data_path}/test_idx.npy", np.asarray(self.test_idx.index))
    #     if target:
    #         target_path = os.makedirs(f"{self.data_path}/{target}", exist_ok=True)
    #         np.save(f"{target_path}/y_train.npy", self.y_train)
    #         np.save(f"{target_path}/y_test.npy", self.y_test)
    #     else:
    #         np.save(f"{self.data_path}/y_train.npy", self.y_train)
    #         np.save(f"{self.data_path}/y_test.npy", self.y_test)
    #     print("Train-test data saved as numpy arrays:\n")
    #     print(os.listdir(self.data_path))

    def save_X_train_test(self):
        np.save(f"{self.data_path}/X_train.npy", np.asarray(self.X_train))
        np.save(f"{self.data_path}/X_test.npy", np.asarray(self.X_test))

    def save_y_train_test(self, target=None):
        if target:
            data_path = os.path.join(self.data_path, target)
        else:
            data_path = self.data_path
        os.makedirs(data_path, exist_ok=True)
        np.save(f"{data_path}/y_train.npy", self.y_train)
        np.save(f"{data_path}/y_test.npy", self.y_test)

    def save_test_index(self, target=None):
        if target:
            idx_path = f"{self.data_path}/{target}/test_idx.npy"
        else:
            idx_path = f"{self.data_path}/test_idx.npy"
        os.makedirs(idx_path, exist_ok=True)
        np.save(idx_path, np.asarray(self.test_idx.index))

    def load_train_test(self, target=None):
        self.X_train, self.X_test = self.load_X_train_test()
        self.y_train, self.y_test = self.load_y_train_test(target=target)
        if self.idx:
            self.test_idx = self.load_test_index(self, target=target, y=None)
            return self.X_train, self.y_train, self.X_test, self.y_test, self.test_idx
        else:
            return self.X_train, self.y_train, self.X_test, self.y_test

    def load_X_train_test(self):
        X_train = np.load(f"{self.data_path}/X_train.npy")
        X_test = np.load(f"{self.data_path}/X_test.npy")
        return X_train, X_test

    def load_y_train_test(self, target=None):
        if target:
            target_path = os.path.join(self.data_path, target)
            y_train = np.load(f"{target_path}/y_train.npy")
            y_test = np.load(f"{target_path}/y_test.npy")
        else:
            y_train = np.load(f"{self.data_path}/y_train.npy")
            y_test = np.load(f"{self.data_path}/y_test.npy")
        return y_train, y_test

    def load_test_index(self, target=None, y=None):
        if target:
            idx_path = f"{self.data_path}/{target}/test_idx.npy"
        else:
            idx_path = f"{self.data_path}/test_idx.npy"
        if os.path.exists(idx_path):
            test_idx = np.load(idx_path, allow_pickle=True)
            if y is None:
                y = self.y_test
            test_idx = pd.DataFrame(
                np.argmax(y, axis=-1),
                index=test_idx,
                columns=[target],
            )
            return test_idx

    # TODO
    def save_compressed(self, y_dict=None, idx_dict=None):
        """Store compressed data to disk"""
        np.savez(f"{self.data_path}/X.npz", X_train=self.X_train, X_test=self.X_test)
        if y_dict:
            np.savez(f"{self.data_path}/y.npz", **y_dict)
        else:
            np.savez(
                f"{self.data_path}/y.npz", y_train=self.y_train, y_test=self.y_test
            )
        if idx_dict:
            np.savez(f"{self.data_path}/idx.npz", **idx_dict)
        else:
            np.savez(f"{self.data_path}/idx.npz", test_idx=self.test_idx)

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
class HstCalData(ArrayOps):
    def __init__(
        self, data_path=".", idx="ipst", targets=["mem_bin", "memory", "wallclock"]
    ):
        super().__init__(data_path=data_path, idx=idx, targets=targets)
        self.y_bin_train = None
        self.y_bin_test = None
        self.y_mem_train = None
        self.y_mem_test = None
        self.y_wall_train = None
        self.y_wall_test = None
        self.bin_test_idx = None
        self.mem_test_idx = None
        self.wall_test_idx = None

    def load_training_data(self):
        self.X_train, self.X_test = self.load_X_train_test()
        self.y_bin_train, self.y_bin_test = self.load_y_train_test(target="mem_bin")
        self.y_mem_train, self.y_mem_test = self.load_y_train_test(target="memory")
        self.y_wall_train, self.y_wall_test = self.load_y_train_test(target="wallclock")
        self.bin_test_idx = self.load_test_index(target="mem_bin", y=self.y_bin_test)
        self.mem_test_idx = self.load_test_index(target="memory", y=self.y_mem_test)
        self.wall_test_idx = self.load_test_index(
            target="wallclock", y=self.y_wall_test
        )
        return self

    def load_compressed(self):
        """Store compressed data to disk"""
        data = np.load(f"{self.data_path}/train_test.npz")

        self.X_train, self.X_test = data["X_train"], data["X_test"]
        self.y_bin_train, self.y_bin_test = data["y_bin_train"], data["y_bin_test"]
        self.y_mem_train, self.y_mem_test = data["y_mem_train"], data["y_mem_test"]
        self.y_wall_train, self.y_wall_test = data["y_wall_train"], data["y_wall_test"]
        data.close()

        idx = np.load(f"{self.data_path}/idx.npz")
        self.bin_test_idx = idx["bin_test_idx"]
        self.mem_test_idx = idx["mem_test_idx"]
        self.wall_test_idx = idx["wall_test_idx"]
        idx.close()
        return self


# TODO
class HstSvmData(ArrayOps):
    def __init__(self, data_path=".", idx="index", targets=["label"]):
        super().__init__(data_path=data_path, idx=idx, targets=targets)

    def save_ensemble_data(self):
        X_train_mlp = np.asarray(self.X_train[0])
        X_train_img = np.asarray(self.X_train[1])
        X_test_mlp = np.asarray(self.X_test[0])
        X_test_img = np.asarray(self.X_test[1])
        y_train = np.asarray(self.y_train)
        y_test = np.asarray(self.y_test)
        test_idx = np.asarray(self.test_idx)
        arrays = [
            X_train_mlp,
            X_train_img,
            X_test_mlp,
            X_test_img,
            y_train,
            y_test,
            test_idx,
        ]
        names = [
            "X_train_mlp",
            "X_train_img",
            "X_test_mlp",
            "X_test_img",
            "y_train",
            "y_test",
            "test_idx",
        ]
        self.save_compressed(arrays, names)

    def load_ensemble_data(self):
        X_train_mlp, X_train_img = np.load(
            f"{self.data_path}/X_train_mlp.npz"
        ), np.load(f"{self.data_path}/X_train_img.npz")
        X_test_mlp, X_test_img = np.load(f"{self.data_path}/X_test_mlp.npz"), np.load(
            f"{self.data_path}/X_test_img.npz"
        )
        self.X_train = [X_train_mlp["arr_0"], X_train_img["arr_0"]]
        self.X_test = [X_test_mlp["arr_0"], X_test_img["arr_0"]]
        self.y_train = np.load(f"{self.data_path}/y_train.npz")["arr_0"]
        self.y_test = np.load(f"{self.data_path}/y_test.npz")["arr_0"]
        self.test_idx = np.load(f"{self.data_path}/test_idx.npz")["arr_0"]


"""Image Ops"""


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
