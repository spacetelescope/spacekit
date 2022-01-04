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
clf = MemoryClassifier(data.X_train, data.y_bin_train, data.X_test, data.y_bin_test)
clf.build_mlp()
clf.fit()
clf.test_idx = data.bin_test_idx
bCom = ComputeMulti(builder=clf, res_path=f"{res_path}/mem_bin")
bCom.calculate_multi()
bin_outputs = bCom.make_outputs()

## 4b: Memory Regressor
mem = MemoryRegressor(data.X_train, data.y_mem_train, data.X_test, data.y_mem_test)
mem.build_mlp()
mem.fit()
mem.test_idx = data.mem_test_idx
mCom = ComputeRegressor(builder=mem, res_path=f"{res_path}/memory")
mCom.calculate_results()
mem_outputs = mCom.make_outputs()

"""
## once saved, these can easily be re-loaded later in a separate session:
# mCom2 = ComputeRegressor(res_path=f"{res_path}/memory")
# res = mCom2.upload()
# mCom2.load_results(res)
"""

## 4c: Wallclock Regressor
wall = WallclockRegressor(data.X_train, data.y_wall_train, data.X_test, data.y_wall_test)
wall.build_mlp()
wall.fit_params(batch_size=64, epochs=300, lr=1e-4)
wall.fit()
wall.test_idx = data.wall_test_idx
wCom = ComputeRegressor(builder=wall, res_path=f"{res_path}/wallclock")
wCom.calculate_results()
wall_outputs = wCom.make_outputs()

# # 5 - export/save
# # save test index to disk (useful for analyzing FN/FP)
# np.save(f"{data_path}/test_idx.npy", np.asarray(data.test_idx.index))
