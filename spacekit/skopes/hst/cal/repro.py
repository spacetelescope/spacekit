from spacekit.datasets.hst_cal import calcloud_data, calcloud_uri
from spacekit.extractor.scrape import WebScraper
from spacekit.analyzer.scan import CalScanner, import_dataset
from spacekit.preprocessor.scrub import ScrubCal
import numpy as np

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

# 4 - build & train models; compute results

## 4a: Memory Bin Classifier
from spacekit.builder.networks import MemoryClassifier
clf = MemoryClassifier(data.X_train, data.y_bin_train, data.X_test, data.y_bin_test)
clf.build_mlp()
clf.fit()


## 4b: Memory Regressor
from spacekit.builder.networks import MemoryRegressor
mem = MemoryRegressor(data.X_train, data.y_mem_train, data.X_test, data.y_mem_test)
mem.build_mlp()
mem.fit()

from spacekit.analyzer.compute import ComputeRegressor
mem.test_idx = data.mem_test_idx
mCom = ComputeRegressor(builder=mem, res_path=f"{data_path}/results/memory")
#mCom.inputs(mem.model, mem.history, mem.X_train, mem.y_train, mem.X_test, mem.y_test, data.mem_test_idx)


## 4c: Wallclock Regressor
from spacekit.builder.networks import WallclockRegressor
wall = WallclockRegressor(data.X_train, data.y_wall_train, data.X_test, data.y_wall_test)
wall.build_mlp()
wall.fit()


# 5 - export/save
# save test index to disk (useful for analyzing FN/FP)
np.save(f"{data_path}/test_idx.npy", np.asarray(data.test_idx.index))

