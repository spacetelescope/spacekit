
# if __name__ == '__main__':
#     radio = Radio()
#     radio.mast_aws()
#     analyzer = Analyzer()
#     analyzer.atomic_vector_plotter()
#     analyzer.make_specgram()
#     analyzer.planet_hunter()
#     transformer = Transformer()
#     transformer.hypersonic_pliers()
#     transformer.thermo_fusion_chisel()
#     transformer.babel_fish_dispenser()
#     transformer.fast_fourier()
#     builder = Builder()
#     builder.build_cnn()
#     builder.fit_cnn()
#     computer = Computer()
#     computer.get_preds()
#     computer.fnfp()
#     computer.keras_history()
#     computer.fusion_matrix()
#     computer.roc_plots()
#     computer.compute()

import os,sys
import wget # wget-3.2
from wget import bar_thermometer
import zipfile
import numpy as np
import spacekit
from spacekit.transformer import Transformer
from spacekit.builder import Builder

def download_data():
    HOME = os.path.curdir
    TRAIN_URL = 'https://github.com/alphasentaurii/starskope/raw/master/DATA/exoTrain.csv.zip'
    TEST_URL = 'https://github.com/alphasentaurii/starskope/raw/master/DATA/exoTest.csv.zip'
    try:
        os.makedirs('data',exist_ok=False)
    except:
        print('Found existing DATA path.')
    DATA = os.path.abspath(HOME+'/data/')
    TRAIN_FILE = wget.download(TRAIN_URL, out=DATA, bar=bar_thermometer)
    print('\n')
    TEST_FILE = wget.download(TEST_URL, out=DATA, bar=bar_thermometer)
    print('\n')
    files = os.listdir(DATA)
    train_zip = os.path.join(DATA,files[0])
    test_zip = os.path.join(DATA,files[1])
    return train_zip, test_zip

def set_params(argv):
    print(len(argv))
    if len(argv) < 2:
        learning_rate=float(1e-5)
        epochs=int(5)
    else:
        learning_rate = float(argv[1])
        epochs = int(argv[2])
    return learning_rate, epochs

class Prep:
    def __init__(self):
        HOME = os.path.curdir
        DATA = os.path.abspath(HOME+'/data/')
        self.HOME = HOME
        self.DATA = DATA
        T = Transformer()
        self.T = T

    def unzip(self, train_zip, test_zip):
        DATA = self.DATA
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(self.DATA)
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall(self.DATA)
        print('Data Extraction Successful')
        os.remove(train_zip)
        os.remove(test_zip)
        os.listdir(DATA)
        #'/exoTrain.csv', '/exoTest.csv'
        return

    def split_data(self, train, test):
        DATA = self.DATA
        T = self.T
        print('Train-Test Split Successful')
        X_train, X_test, y_train, y_test = T.hypersonic_pliers(DATA+train, DATA+test)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        T = self.T
        print('Data Scaled to Zero Mean and Unit Variance')
        X_train, X_test = T.thermo_fusion_chisel(X_train, X_test)
        return X_train, X_test

    def add_filter(self, X_train, X_test):
        T = self.T
        print('Noise filter added!')
        X_train, X_test = T.babel_fish_dispenser(X_train, X_test)
        return X_train, X_test

class Launch:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        builder = Builder(X_train, X_test, y_train, y_test)
        self.builder = builder

    def deploy(self):
        builder = self.builder
        cnn = builder.build_cnn()
        return cnn

    def takeoff(self, cnn):
        builder = self.builder
        history = builder.fit_cnn(cnn)

if __name__ == '__main__':
    import spacekit
    train_zip, test_zip = download_data()
    prep = Prep()
    prep.unzip(train_zip, test_zip)
    
    X_train, X_test, y_train, y_test = prep.split_data('/exoTrain.csv', '/exoTest.csv')
    X_train, X_test = prep.scale_data(X_train, X_test)
    X_train, X_test = prep.add_filter(X_train, X_test)

    learning_rate, epochs = set_params(sys.argv)
    launch = Launch(X_train, X_test, y_train, y_test)
    cnn = launch.deploy()
    history = launch.takeoff(cnn)
