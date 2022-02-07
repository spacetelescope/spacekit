from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from app import app
from spacekit.dashboard.svm import eda, eval, pred
from spacekit.dashboard.svm.config import svm, hst, NN, tx_file


url_bar_and_content_div = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)
# index layout
index_layout = html.Div(
    children=[
        html.Br(),
        html.H1("SPACEKIT", style={"padding": 15}),
        html.H2("Dashboard"),
        html.Div(
            children=[
                html.Div("Model Performance + Statistical Analysis"),
                html.Div("for HST Single Visit Mosaic"),
                html.Div("Image Alignment Classification"),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Br(),
                dcc.Link("Evaluation", href="/eval"),
                html.Br(),
                dcc.Link("Analysis", href="/eda"),
                html.Br(),
                dcc.Link("Prediction", href="/pred"),
            ]
        ),
    ],
    style={
        "backgroundColor": "#1b1f34",
        "color": "white",
        "textAlign": "center",
        "width": "80%",
        "display": "inline-block",
        "float": "center",
        "padding": "10%",
        "height": "99vh",
    },
)
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div(
    [url_bar_and_content_div, index_layout, eval.layout, eda.layout, pred.layout]
)



@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return index_layout # home.layout
    elif pathname == "/eval":
        return eval.layout
    elif pathname == "/eda":
        return eda.layout
    elif pathname == "/pred":
        return pred.layout
    else:
        return "Woops! 404 :("

# Page 1 EVAL callbacks
# KERAS CALLBACK
@app.callback(
    [Output("keras-acc", "figure"), Output("keras-loss", "figure")],
    Input("keras-picker", "value"),
)
def update_keras(selected_version):
    return svm.keras[selected_version]


# ROC AUC CALLBACK
@app.callback(
    [Output("roc-auc", "figure"), Output("precision-recall-fig", "figure")],
    Input("rocauc-picker", "value"),
)
def update_roc_auc(selected_version):
    return svm.roc[selected_version]


# CMX Callback
@app.callback(Output("confusion-matrix", "figure"), Input("cmx-type", "value"))
def update_cmx(cmx_type):
    v = list(svm.mega.keys())[-1]
    return svm.triple_cmx(svm.cmx[cmx_type], cmx_type, classes=["2GB", "8GB", "16GB", "64GB"])

# TODO
# SCATTER CALLBACK
@app.callback(
    [
        Output("hrc-scatter", "figure"),
        Output("ir-scatter", "figure"),
        Output("sbc-scatter", "figure"),
        Output("uvis-scatter", "figure"),
        Output("wfc-scatter", "figure"),
    ],
    [Input("selected-scatter", "value")],
)
def update_scatter(selected_scatter):
    # hst.scatter = [rms_scatter, source_scatter]
    scatter_figs = {"rms-ra-dec": hst.scatter[0], "point-segment": hst.scatter[1]}
    return scatter_figs[selected_scatter]


# BARPLOT CALLBACK
@app.callback(
    [
        Output("hrc-bars", "figure"),
        Output("ir-bars", "figure"),
        Output("sbc-bars", "figure"),
        Output("uvis-bars", "figure"),
        Output("wfc-bars", "figure"),
    ],
    [Input("selected-barplot", "value")],
)
def update_barplot(selected_barplot):
    bar_figs = {
        "rms_ra": hst.bar[0],
        "rms_dec": hst.bar[1],
        "gaia": hst.bar[2],
        "nmatches": hst.bar[3],
        "numexp": hst.bar[4],
    }
    return bar_figs[selected_barplot]


# hst.kde = [kde_rms, kde_targ, kde_norm]
# kde_rms = ["ra_dec"]
# kde_targ = ["rms_ra", "rms_dec", "gaia", "nmatches", "numexp"]
# kde_norm = ["rms_ra", "rms_dec", "gaia", "nmatches", "numexp"]




# # Box Plots
# @app.callback(
#     [Output("point", "figure"), Output("segment", "figure")],
#     Input("continuous-vars", "value"),
# )
# def update_continuous(raw_norm):
#     if raw_norm == "raw":
#         vars = ["point", "segment"]
#     elif raw_norm == "norm":
#         vars = ["point_scl", "segment_scl"]
#     continuous_figs = hst.make_continuous_figs(vars)
#     return continuous_figs


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True, dev_tools_prune_errors=False)
    #app.run_server(debug=True)
