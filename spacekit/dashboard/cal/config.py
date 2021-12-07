from spacekit.analyzer.explore import HstCalPlots
from spacekit.analyzer.scan import CalScanner, import_dataset

# Find available datasets and load most recent (default)
cal = CalScanner(perimeter=f"data/20??-*-*-*", primary=-1)

cal.data = cal.select_dataset() # "data/2021-11-04-1636048291/latest.csv"

df = import_dataset(
    filename=cal.data, kwargs=dict(index_col="ipst"), 
    decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}
    )
hst = HstCalPlots(df).df_by_instr()
