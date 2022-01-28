from spacekit.analyzer.explore import HstSvmPlots
from spacekit.analyzer.scan import SvmScanner, import_dataset

mega = SvmScanner(perimeter="data/20??-*-*-*", primary=-1)
mega.scan_results()

df = mega.load_dataframe()
# df = import_dataset(
#     filename=mega.data,
#     kwargs=dict(index_col="index"),
#     decoder_key={"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}},
# )

hst = HstSvmPlots(df, group="det")
hst.df_by_detector()
hst.draw_plots()
