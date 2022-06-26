from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from app import app
from spacekit.dashboard.svm import eda, eval, pred
from spacekit.dashboard.svm.config import svm, hst, images
from spacekit.analyzer.explore import SVMPreviews
from spacekit.generator.augment import augment_image
import numpy as np

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
        return index_layout  # home.layout
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
    return svm.triple_cmx(svm.cmx[cmx_type], cmx_type)


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
    # {'rms_ra_dec': rms_scatter, 'point_segment': source_scatter}
    return hst.scatter[selected_scatter]


# BARPLOT CALLBACK
@app.callback(
    Output("bar-group", "figure"),
    [Input("selected-barplot", "value")],
)
def update_barplot(selected_barplot):
    return hst.bar[selected_barplot]


# KDE CALLBACK
@app.callback(
    [
        Output("kde-targ", "figure"),
        Output("kde-norm", "figure"),
        Output("kde-rms", "figure"),
    ],
    [Input("selected-kde", "value")],
)
def update_kde(selected_kde):
    return [
        hst.kde["targ"][selected_kde],
        hst.kde["norm"][selected_kde],
        hst.kde["rms"],
    ]


@app.callback(Output("image-graph", "figure"), Input("selected-image", "value"))
def update_image_previews(selected_image):
    (idx, X, y) = images
    img_num = np.where(idx == selected_image)
    x = X[img_num].reshape(3, 128, 128, 3)
    label = y[img_num]
    x_prime = augment_image(x)
    previews = SVMPreviews(X, y, x_prime, y)


# 3D Scatter
# @app.callback(
#     Output("scatter-3d", "figure"),
#     Input("point-slider", "value")
# )
# def update_3d_scatter_point(p_range):
#     p_low, p_high = p_range
#     p_mask = (hst.df.point > p_low) & (hst.df.point < p_high)
#     masked = hst.df[p_mask]
#     fig = hst.scatter3d('point', 'segment', 'gaia', mask=masked, width=1000, height=1000)
#     return fig
# @app.callback(
#     Output("scatter-3d", "figure"),
#     [Input("point-slider", "value"),
#     Input("segment-slider", "value"),
#     Input("gaia-slider", "value")]
# )
# def update_3d_scatter_point(p_range, s_range, g_range):
#     p_low, p_high = p_range
#     p_mask = (hst.df.point > p_low) & (hst.df.point < p_high)
#     s_low, s_high = s_range
#     s_mask = (hst.df.segment > s_low) & (hst.df.segment < s_high)
#     g_low, g_high = g_range
#     g_mask = (hst.df.gaia > g_low) & (hst.df.gaia < g_high)
#     masked = hst.df[p_mask & s_mask & g_mask]
#     fig = hst.scatter3d('point', 'segment', 'gaia', mask=masked, width=1000, height=1000)
#     return fig


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
    # app.run_server(debug=True)
