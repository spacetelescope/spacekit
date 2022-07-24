from spacekit.analyzer.explore import HstCalPlots
from spacekit.analyzer.scan import CalScanner
from spacekit.builder.architect import Builder

# Find available datasets and load most recent (default)
cal = CalScanner(perimeter="data/20??-*-*-*", primary=-1)
cal.scan_results()
# Initialize EVAL
cal.make_barplots()  # scores, acc_fig, loss_fig, acc_loss_fig
cal.make_clf_plots(target="mem_bin")  # keras, roc, pr, cmx

# Initialize EDA
cal.load_dataframe()
hst = HstCalPlots(cal.df)  # .df_by_instr()
hst.draw_plots()
# Initialize PRED
selection = cal.datapaths[cal.primary]
model_path = f"{selection}/models"
tx_file = f"{model_path}/tx_data.json"
ipsts = cal.df.index.values


clf = Builder(
    blueprint="memory_classifier", model_path=model_path + "/mem_clf"
).load_saved_model()

mem_reg = Builder(
    blueprint="memory_regression", model_path=model_path + "/mem_reg"
).load_saved_model()

wall_reg = Builder(
    blueprint="wallclock_regression", model_path=model_path + "/wall_reg"
).load_saved_model()


global NN
NN = {"mem_clf": clf, "mem_reg": mem_reg, "wall_reg": wall_reg}
