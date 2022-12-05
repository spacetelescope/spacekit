from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from argparse import ArgumentParser
from app import app
from spacekit.dashboard.svm import eda, eval, pred
from spacekit.dashboard.svm.config import svm, hst, images  # NN
import numpy as np

from spacekit.analyzer.explore import SVMPreviews


(idx, X, y) = images
global imps
imps = SVMPreviews(X, labels=y, names=idx)

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


# PAGE 2 EDA CALLBACKS
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

# PAGE 3 PRED CALLBACKS


def select_images(selected_image):
    img_num = np.where(imps.names == selected_image)
    Xi = imps.select_image_from_array(i=img_num)
    return Xi


@app.callback(
    [Output("original-image", "figure"), Output("augment-button-state", "disabled")],
    Input("preview-button-state", "n_clicks"),
    State("image-state", "value"),
)
def update_image_previews(n_clicks, selected_image):
    if n_clicks == 0:
        raise PreventUpdate
    Xi = select_images(selected_image)
    fig = imps.preview_image(Xi)
    disabled = False
    return [fig, disabled]


@app.callback(
    Output("augmented-image", "figure"),
    Input("augment-button-state", "n_clicks"),
    State("image-state", "value"),
)
def preview_augmented_image(n_clicks, selected_image):
    if n_clicks == 0:
        raise PreventUpdate
    Xp = select_images(selected_image)
    fig = imps.preview_image(Xp, aug=True)
    return fig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--env", type=str, choices=["prod", "dev"], default="dev")
    args = parser.parse_args()
    host, port = args.host, args.port
    if args.env == "dev":
        app.run_server(host=host, port=port, debug=True, dev_tools_prune_errors=False)
    else:
        app.run_server(host="0.0.0.0", port=8050)
