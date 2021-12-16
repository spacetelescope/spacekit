import os
import pandas as pd
import numpy as np
from zipfile import ZipFile
import numpy as np
from keras.preprocessing import image
from spacekit.extractor.scrape import JsonScraper

class FileOps:
    def __init__(self):
        pass

class ArrayIO(FileOps):
    def __init__(self):
        super().__init__(self)

    """Pandas/Numpy File ops"""
    def load_train_test(data_path=".", idx=True, target="mem_bin"):
        X_train, y_train = np.load(f"{data_path}/X_train.npy"), np.load(
            f"{data_path}/y_train.npy"
        )
        X_test, y_test = np.load(f"{data_path}/X_test.npy"), np.load(
            f"{data_path}/y_test.npy"
        )
        if idx is True:
            test_idx = np.load(f"{data_path}/test_idx.npy", allow_pickle=True)
            if target:
                test_idx = pd.DataFrame(
                    np.argmax(y_test, axis=-1), index=idx, columns=[target]
                )
            return X_train, y_train, X_test, y_test, test_idx
        else:
            return X_train, y_train, X_test, y_test


    def save_train_test(X_train, X_test, y_train, y_test, test_idx, dpath):
        np.save(f"{dpath}/X_train.npy", np.asarray(X_train))
        np.save(f"{dpath}/X_test.npy", np.asarray(X_test))
        np.save(f"{dpath}/y_train.npy", y_train)
        np.save(f"{dpath}/y_test.npy", y_test)
        np.save(f"{dpath}/test_idx.npy", np.asarray(test_idx.index))
        print("Train-test data saved as numpy arrays:\n")
        print(os.listdir(dpath))

class Hdf5IO(FileOps):
    def __init__(self, h5_file=None, search_path="", patterns=["*_total*_svm_*.json"],  crpt=0, save_file_as="svm_data", outpath=None):
        super().__init__(self)
        self.h5_file = h5_file
        self.data = None
        self.patterns = patterns
        self.search_path = search_path
        self.crpt = crpt
        self.save_file_as = save_file_as
        self.outpath = outpath

    def load_h5_file(self):
        if not self.h5_file.endswith(".h5"):
            self.h5_file += ".h5"
        if os.path.exists(self.h5_file):
            with pd.HDFStore(self.h5_file) as store:
                self.data = store["mydata"]
                print(f"Dataframe created: {self.data.shape}")
        else:
            errmsg = "HDF5 file {} not found!".format(self.h5_file)
            print(errmsg)
            raise Exception(errmsg)
        return self.data

    def make_h5_file(self):
        print("\n*** Starting JSON Harvest ***")
        filename = self.save_file_as.split(".")[0]
        if self.outpath is None:
            self.outpath = os.getcwd()
        self.h5_file = os.path.join(self.outpath, filename)
        jsc = JsonScraper(search_path=self.search_path, search_patterns=self.patterns, h5_filename=self.h5_file, crpt=self.crpt)
        self.h5_file = jsc.h5_file
        self.data = jsc.data
        return self


"""Image Ops"""
class ImageIO(FileOps):
    def __init__(self):
        super().__init__(self)

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


# Images saved as numpy arrays:
#     num_train_samples = 50000

#     x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
#     y_train = np.empty((num_train_samples,), dtype='uint8')

#     for i in range(1, 6):
#         fpath = os.path.join(path, 'data_batch_' + str(i))
#         (x_train[(i - 1) * 10000:i * 10000, :, :, :],
#         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

#     fpath = os.path.join(path, 'test_batch')
#     x_test, y_test = load_batch(fpath)

#     y_train = np.reshape(y_train, (len(y_train), 1))
#     y_test = np.reshape(y_test, (len(y_test), 1))

#     if backend.image_data_format() == 'channels_last':
#         x_train = x_train.transpose(0, 2, 3, 1)
#         x_test = x_test.transpose(0, 2, 3, 1)

#     x_test = x_test.astype(x_train.dtype)
#     y_test = y_test.astype(y_train.dtype)

#     return (x_train, y_train), (x_test, y_test)