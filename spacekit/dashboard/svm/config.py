from spacekit.analyzer.explore import SingleVisitMosaic
from spacekit.analyzer.scan import MegaScanner, import_dataset

mega = MegaScanner(perimeter="data/20??-*-*-*", primary=-1)

df = import_dataset(
    filename=mega.data,
    kwargs=dict(index_col="index"),
    decoder_key={"det": {0: "hrc", 1: "ir", 2: "sbc", 3: "uvis", 4: "wfc"}},
)

hst = SingleVisitMosaic(df, group="det")
hst.df_by_detector()
hst.draw_plots()
