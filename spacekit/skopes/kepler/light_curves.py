import os
from spacekit.preprocessor.transform import (
    hypersonic_pliers,
    thermo_fusion_chisel,
    babel_fish_dispenser,
)
from spacekit.builder.networks import Cnn2dBuilder
from spacekit.datasets.k2_exo import k2_uri, k2_data
from spacekit.extractor.scrape import WebScraper


class LaunchK2:
    def __init__(self, fpaths):
        self.fpaths = fpaths
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.builder = None
        self.history = None

    def launch_prep(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.X_train, self.X_test = self.scale_data()
        self.X_train, self.X_test = self.add_filter()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def split_data(self):
        print("Splitting train-test feature and target data...")
        for fpath in self.fpaths:
            if fpath.endswith("Train"):
                train = fpath
            else:
                test = fpath
        self.X_train, self.X_test, self.y_train, self.y_test = hypersonic_pliers(
            train, test
        )
        print("Data split successful")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_data(self):
        print("Scaling data to Zero Mean and Unit Variance...")
        self.X_train, self.X_test = thermo_fusion_chisel(self.X_train, self.X_test)
        print("Data scaling successful.")
        return self.X_train, self.X_test

    def add_filter(self):
        print("Adding noise filter...")
        self.X_train, self.X_test = babel_fish_dispenser(self.X_train, self.X_test)
        print("Noise filter added successfully.")
        return self.X_train, self.X_test

    def deploy(self):
        self.builder = Cnn2dBuilder(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.builder.build_cnn()
        return self.builder

    def takeoff(self):
        self.history = self.builder.batch_fit()


if __name__ == "__main__":
    home = os.getcwd()
    data = os.path.join(home, "data")
    print("Extracting data...")
    fpaths = WebScraper(k2_uri, k2_data).scrape_repo()
    print("Data extraction successful.")
    k2 = LaunchK2(fpaths)
    k2.launch_prep()
    k2.builder = k2.deploy()
    k2.history = k2.takeoff()
