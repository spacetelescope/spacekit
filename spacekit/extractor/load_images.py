import os
from zipfile import ZipFile
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import time

from spacekit.analyzer.track import stopwatch


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
