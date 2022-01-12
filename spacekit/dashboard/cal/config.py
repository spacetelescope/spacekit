from spacekit.analyzer.explore import HstCalPlots
from spacekit.analyzer.scan import CalScanner, import_dataset
from spacekit.dashboard.cal.nodegraph import get_model

# Find available datasets and load most recent (default)
cal = CalScanner(perimeter="data/20??-*-*-*", primary=-1)

cal.scan_results()

# cal.compare_scores()
# self.acc_fig = None  # self.acc_bars()
# self.loss_fig = None  # self.loss_bars()
# self.acc_loss_figs = None  # self.acc_loss_subplots()

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
clf = get_model("mem_clf", model_path=model_path)

global mem_reg
mem_reg = get_model("mem_reg", model_path=model_path)

global wall_reg
wall_reg = get_model("wall_reg", model_path=model_path)

global tx_file
tx_file = f"{model_path}/pt_transform"

global NN
NN = {"mem_clf": clf, "mem_reg": mem_reg, "wall_reg": wall_reg}
