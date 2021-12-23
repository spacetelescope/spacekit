# TODO: eda for regression test data (MLP data)

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from spacekit.dashboard.svm.app import app
from spacekit.dashboard.svm.config import hst

layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Br(),
                dcc.Link("Home", href="/layouts/home"),
                html.Br(),
                dcc.Link("Evaluation", href="/layouts/eval"),
                html.Br(),
                dcc.Link("Prediction", href="/layouts/pred"),
                html.Br(),
            ]
        ),
        # FEATURE SCATTERPLOTS
        html.Div(
            children=[
                html.Div(
                    [
                        dcc.Dropdown(
                            id="selected-scatter",
                            options=[
                                {"label": f, "value": f}
                                for f in ["rms-ra-dec", "point-segment"]
                            ],
                            value="rms-ra-dec",
                        )
                    ],
                    style={
                        "width": "20%",
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="hrc-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="ir-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="sbc-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="uvis-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="wfc3-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
                ),
            ]
        ),
        # FEATURE BARPLOTS
        html.Div(
            children=[
                html.Div(
                    [
                        dcc.Dropdown(
                            id="selected-barplot",
                            options=[
                                {"label": f, "value": f}
                                for f in [
                                    "rms_ra",
                                    "rms_dec",
                                    "gaia",
                                    "nmatches",
                                    "numexp",
                                ]
                            ],
                            value="rms-ra",
                        )
                    ],
                    style={
                        "width": "20%",
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="hrc-bars",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="ir-bars",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="sbc-bars",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="uvis-bars",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="wfc3-bars",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
                ),
            ]
        ),
    ]
)


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
