import os
from spacekit.preprocessor.transform import (
    hypersonic_pliers,
    thermo_fusion_chisel,
    babel_fish_dispenser,
)
from spacekit.builder.architect import BuilderCNN2D
from spacekit.datasets.meta import k2 as k2meta
from spacekit.extractor.scrape import WebScraper

def downloads_exist(scraper, k2_meta):
    base_path = os.path.join(scraper.cache_dir, scraper.cache_subdir)
    filepaths = []
    for k, v in k2_meta.items():
        fpath = os.path.join(base_path, v['key'])
        filepaths.append(fpath)
    for fp in filepaths:
        if not os.path.exists(fp):
            return []
    print("Found existing datasets, skipping download.")
    return filepaths


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
        self.split_data()
        self.scale_data()
        self.add_filter()

    def split_data(self):
        print("Splitting train-test feature and target data...")
        for fpath in self.fpaths:
            if "Train" in fpath:
                train = fpath
            else:
                test = fpath
        self.X_train, self.X_test, self.y_train, self.y_test = hypersonic_pliers(
            train, test, subtract_y=1.0, reshape=True
        )
        print("Data split successful")

    def scale_data(self):
        print("Scaling data to Zero Mean and Unit Variance...")
        self.X_train, self.X_test = thermo_fusion_chisel(self.X_train, self.X_test)
        print("Data scaling successful.")

    def add_filter(self):
        print("Adding noise filter...")
        self.X_train, self.X_test = babel_fish_dispenser(self.X_train, self.X_test)
        print("Noise filter added successfully.")

    def deploy(self):
        self.builder = BuilderCNN2D(
            X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test
        )
        self.builder.build()

    def takeoff(self):
        self.history = self.builder.batch_fit()


if __name__ == "__main__":
    print("Extracting data...")
    scraper = WebScraper(k2meta['uri'], k2meta['data'])
    scraper.fpaths = downloads_exist(scraper, k2meta['data'])
    if not scraper.fpaths:
        scraper.scrape()
        print("Data extraction successful.")
    k2 = LaunchK2(scraper.fpaths)
    k2.launch_prep()
    k2.deploy()
    k2.takeoff()
