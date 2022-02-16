from pytest import mark


@mark.analyzer
@mark.explore
def make_scatter_plots(explorer):
    x, y = explorer.feature_list[:2]
    figs = explorer.make_scatter_figs(x, y)
    assert len(figs) > 0
