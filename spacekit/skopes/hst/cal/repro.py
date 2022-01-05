from spacekit.datasets.hst_cal import calcloud_data, calcloud_uri
from spacekit.extractor.scrape import WebScraper
from spacekit.analyzer.scan import CalScanner, import_dataset
from spacekit.preprocessor.scrub import ScrubCal
from spacekit.builder.networks import MemoryClassifier, MemoryRegressor, WallclockRegressor
from spacekit.analyzer.compute import ComputeMulti, ComputeRegressor

# 1 - get data archives (job data, models, results)
fpaths = WebScraper(calcloud_uri, calcloud_data).scrape_repo()

# 2 - import data
cal = CalScanner(perimeter="data/20??-*-*-*", primary=-1)

df = import_dataset(
    filename=cal.data,
    kwargs=dict(index_col="ipst"),
    decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
)

# 3 - data preprocessing
data = ScrubCal(df).prep_data()
selection = cal.datapaths[cal.primary]
data_path = f"{selection}/data"
res_path = f"{selection}/results"

# 4 - build & train models; compute results
## 4a: Memory Bin Classifier
clf = MemoryClassifier(data.X_train, data.y_bin_train, data.X_test, data.y_bin_test, test_idx=data.bin_test_idx)
clf.build_mlp()
clf.fit()
bCom = ComputeMulti(builder=clf, res_path=f"{res_path}/mem_bin")
bCom.calculate_multi()
bCom.make_outputs()

"""
## to load results from disk in a separate session (for plotting, etc):
# bcom2 = ComputeMulti(res_path=f"{res_path}/mem_bin")
# bin_out = bcom2.upload()
# bcom2.load_results(bin_out)
"""

## 4b: Memory Regressor
mem = MemoryRegressor(data.X_train, data.y_mem_train, data.X_test, data.y_mem_test, test_idx=data.mem_test_idx)
mem.build_mlp()
mem.fit() # using default fit params
mCom = ComputeRegressor(builder=mem, res_path=f"{res_path}/memory")
mCom.calculate_results()
mCom.make_outputs()

"""
## to load results from disk in a separate session (for plotting, etc):
# mcom2 = ComputeRegressor(res_path=f"{res_path}/memory")
# mem_out = mcom2.upload()
# mcom2.load_results(mem_out)
"""

## 4c: Wallclock Regressor
wall = WallclockRegressor(data.X_train, data.y_wall_train, data.X_test, data.y_wall_test, test_idx=data.wall_test_idx)
wall.build_mlp()
wall.fit_params(batch_size=64, epochs=300)
wall.fit()
wCom = ComputeRegressor(builder=wall, res_path=f"{res_path}/wallclock")
wCom.calculate_results()
wCom.make_outputs()

"""
## to load results from disk in a separate session (for plotting, etc):
# wcom2 = ComputeRegressor(res_path=f"{res_path}/wallclock")
# wall_out = wcom2.upload()
# wcom2.load_results(wall_out)
"""

