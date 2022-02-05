from spacekit.analyzer.explore import HstCalPlots
from spacekit.analyzer.scan import CalScanner
from spacekit.builder.architect import Builder

# Find available datasets and load most recent (default)
cal = CalScanner(perimeter="data/20??-*-*-*", primary=-1)
cal.scan_results()
cal.load_dataframe()
cal.compare_scores()
cal.acc_fig = cal.accuracy_bars()
cal.loss_fig = cal.loss_bars()
cal.acc_loss_subplots()

hst = HstCalPlots(cal.df).df_by_instr()

selection = cal.datapaths[cal.primary]
model_path = f"{selection}/models"

global clf
clf = Builder(
    blueprint="memory_classifier", model_path=model_path + "/mem_clf"
).load_saved_model()

global mem_reg
# mem_reg = get_model("mem_reg", model_path=model_path)
mem_reg = Builder(
    blueprint="memory_regression", model_path=model_path + "/mem_reg"
).load_saved_model()

global wall_reg
# wall_reg = get_model("wall_reg", model_path=model_path)
wall_reg = Builder(
    blueprint="wallclock_regression", model_path=model_path + "/wall_reg"
).load_saved_model()

global tx_file
tx_file = f"{model_path}/pt_transform"

global NN
NN = {"mem_clf": clf, "mem_reg": mem_reg, "wall_reg": wall_reg}
