from spacekit.builder.architect import Builder
from spacekit.analyzer.explore import HstSvmPlots
from spacekit.analyzer.scan import SvmScanner

# Find available datasets and load most recent (default)
svm = SvmScanner(perimeter="data/20??-*-*-*", primary=-1)
svm.scan_results()
# Initialize EVAL
svm.make_barplots()
svm.make_clf_plots(target="test")

# Initialize EDA
svm.load_dataframe()
hst = HstSvmPlots(svm.df, group="det", show=False, save_html=None)
hst.draw_plots()
# hst.make_svm_scatterplots()

selection = svm.datapaths[svm.primary]
model_path = f"{selection}/models"
global ens
ens = Builder(blueprint="ensemble", model_path=model_path + "/ensembleSVM")
ens.load_saved_model()

global tx_file
tx_file = f"{model_path}/tx_data"

global NN
NN = {"ens": ens.model}
