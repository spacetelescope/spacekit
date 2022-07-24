from spacekit.builder.architect import Builder
from spacekit.analyzer.explore import HstSvmPlots
from spacekit.analyzer.scan import SvmScanner
from spacekit.extractor.load import ImageIO

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

ens = Builder(blueprint="ensemble", model_path=model_path + "/ensembleSVM")
ens.load_saved_model()

tx_file = f"{model_path}/tx_data.json"
NN = {"ens": ens.model}

images = ImageIO("data/test_images.npz").load_npz()


# ['hst_9454_11_acs_hrc_total_j8ff11',
#  'hst_8992_53_acs_hrc_total_j8cw53',
# 'hst_11099_04_wfc3_ir_total_ia0m04',
#  'hst_11099_04_wfc3_ir_total_ia0m04_f160w_all_stoc',
#  'hst_10183_03_acs_sbc_total_j8y503',
#  'hst_13483_21_acs_sbc_total_jcdb21',
#  'hst_12062_eh_wfc3_uvis_total_ibeveh',
#  'hst_12109_01_wfc3_uvis_total_ibfn01',
#  'hst_8992_03_acs_wfc_total_j8cw03',
#  'hst_9836_33_acs_wfc_total_j8rq33',
#  'hst_8992_03_acs_wfc_total_j8cw03_f475w_clear2l_sub_stat',
#  'hst_9836_33_acs_wfc_total_j8rq33_clear1s_f250w_sub_stoc']
