import os
from keras.utils.data_utils import get_file
import boto3
# from botocore import Config

#retry_config = Config(retries={"max_attempts": 3})
client = boto3.client("s3")#, config=retry_config)

d0 = {
    "fname" : "2021-11-04-1636048291.zip",
    "hash": "e46777117e7a43e1053cc20609e25f798a564333"
}

d1 = {
    "fname": "2021-10-28-1635457222.zip",
    "hash": "ba0f555bfcdc36f977e92c49344cd5379af0a08b"
    }

d2 = {
    "fname": "2021-08-22-1629663047.zip",
    "hash": "55d1943ef3c90b37124448ac13969bc528b53ae4"
    }

def scrape_web(root=None, key=d0):
    if root is None:
        # tmp
        # account/repo/branch/pkg/module/submodule/data
        root = "https://raw.githubusercontent.com/alphasentaurii/spacekit/dashboard/spacekit/dashboard/cal/data"
        # root = "https://raw.githubusercontent.com/alphasentaurii/spacekit/main/dashboard/cal/data/"
    fname = key["fname"]
    origin = f"{root}/{fname}"
    hash = key["hash"]
    fpath = get_file(
        origin=origin,
        file_hash=hash,
        hash_algorithm="sha256", #auto
        cache_dir="~",
        cache_subdir="data",
        extract=True,
        archive_format="zip"
        )
    if os.path.exists(fpath):
        os.remove(f"data/{fname}")
    return fpath


def get_sample_data():
    keys = [d0, d1, d2]
    fpaths = []
    for key in keys:
        fpath = scrape_web(key=key)
        fpaths.append(fpath)
    return fpaths


def scrape_s3(bucket, results=[]):
    res_keys = {}
    for r in results:
        os.makedirs(f"./data/{r}")
        res_keys[r] = [
            "data/latest.csv", 
            "models/models.zip", 
            "results/mem_bin",
            "results/memory",
            "results/wallclock"
        ]
    err = None
    for prefix, key in res_keys.items():
        obj = f"{bucket}/{prefix}/{key}"
        print("s3://", obj)
        try:
            keypath = f"./data/{prefix}/{key}"
            with open(keypath, "wb") as f:
                client.download_fileobj(bucket, obj, f)
        except Exception as e:
            err = e
            continue
    if err is not None:
        print(err)

# def scrape_web_images():
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

import numpy as np
def load_cal_data():
    X_train, y_train = np.load('X_train.npy'), np.load('y_train.npy')
    X_test, y_test = np.load('X_test.npy'), np.load('y_test.npy')
    return X_train, y_train, X_test, y_test