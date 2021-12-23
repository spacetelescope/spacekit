from spacekit.analyzer.explore import HstCalPlots
from spacekit.analyzer.scan import CalScanner, import_dataset
from spacekit.dashboard.cal.nodegraph import get_model

# Find available datasets and load most recent (default)
cal = CalScanner(perimeter="data/20??-*-*-*", primary=-1)

# cal.data = cal.select_dataset() # "data/2021-11-04-1636048291/latest.csv"

df = import_dataset(
    filename=cal.data,
    kwargs=dict(index_col="ipst"),
    decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}},
)
hst = HstCalPlots(df).df_by_instr()

selection = cal.datapaths[cal.primary]
model_path = f"{selection}/models"

global clf
clf = get_model(f"{model_path}/mem_clf")

global mem_reg
mem_reg = get_model(f"{model_path}/mem_reg")

global wall_reg
wall_reg = get_model(f"{model_path}/wall_reg")

global tx_file
tx_file = f"{model_path}/pt_transform"

global NN
NN = {"mem_clf": clf, "mem_reg": mem_reg, "wall_reg": wall_reg}
