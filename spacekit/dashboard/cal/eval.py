from dash import dcc
from dash import html
from spacekit.dashboard.cal.config import cal


layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Br(),
                dcc.Link("Home", href="/"),
                html.Br(),
                dcc.Link("Analysis", href="/eda"),
                html.Br(),
                dcc.Link("Prediction", href="/pred"),
                html.Br(),
            ]
        ),
        html.Div(
            children=[
                html.H3("Model Performance"),
                # MEMORY CLASSIFIER CHARTS
                html.Div(
                    children=[
                        html.H4(
                            children="Memory Bin Classifier", style={"padding": 10}
                        ),
                        # ACCURACY vs LOSS (BARPLOTS)
                        "Accuracy vs Loss",
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="acc-bars",
                                    figure=cal.acc_fig,
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="loss-bars",
                                    figure=cal.loss_fig,
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                            ]
                        ),
                        # KERAS HISTORY
                        html.P("Keras History", style={"margin": 25}),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="keras-picker",
                                                    options=[
                                                        {"label": str(v), "value": v}
                                                        for v in cal.versions
                                                    ],
                                                    value=cal.versions[-1],
                                                )
                                            ],
                                            style={
                                                "color": "black",
                                                "display": "inline-block",
                                                "float": "center",
                                                "width": 150,
                                            },
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="keras-acc",
                                    figure=cal.keras[cal.versions[-1]][0],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="keras-loss",
                                    figure=cal.keras[cal.versions[-1]][1],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                            ]
                        ),
                        html.P("ROC AUC", style={"margin": 25}),
                        html.P(
                            "Receiver Operator Characteristic", style={"margin": 25}
                        ),
                        html.P("(Area Under the Curve)", style={"margin": 25}),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="rocauc-picker",
                                                    options=[
                                                        {"label": str(v), "value": v}
                                                        for v in cal.versions
                                                    ],
                                                    value=cal.versions[-1],
                                                )
                                            ],
                                            style={
                                                "color": "black",
                                                "display": "inline-block",
                                                "float": "center",
                                                "width": 150,
                                            },
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="roc-auc",
                                    figure=cal.roc[cal.versions[-1]][0],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="precision-recall-fig",
                                    figure=cal.roc[cal.versions[-1]][1],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                            ]
                        ),
                        # CONFUSION MATRIX
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="cmx-type",
                                            options=[
                                                {"label": "counts", "value": "counts"},
                                                {
                                                    "label": "normalized",
                                                    "value": "normalized",
                                                },
                                            ],
                                            value="counts",
                                        )
                                    ],
                                    style={
                                        "color": "black",
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": 150,
                                    },
                                ),
                                dcc.Graph(
                                    id="confusion-matrix",
                                    figure=cal.triple_cmx(cal.cmx["counts"], "counts"),
                                ),
                            ],
                            style={
                                "color": "white",
                                "padding": 50,
                                "display": "inline-block",
                                "width": "80%",
                            },
                        ),
                    ],
                    style={
                        "color": "white",
                        "border": "2px #333 solid",
                        "borderRadius": 5,
                        "margin": 25,
                        "padding": 10,
                    },
                ),
            ]
        ),
    ],
    style={
        "backgroundColor": "#1b1f34",
        "color": "white",
        "textAlign": "center",
        "width": "100%",
        "display": "inline-block",
        "float": "center",
    },
)


# Moved to Index page
# # Page 1 EVAL callbacks
# # KERAS CALLBACK
# @app.callback(
#     [Output("keras-acc", "figure"), Output("keras-loss", "figure")],
#     Input("keras-picker", "value"),
# )
# def update_keras(selected_version):
#     return cal.keras[selected_version]


# # ROC AUC CALLBACK
# @app.callback(
#     [Output("roc-auc", "figure"), Output("precision-recall-fig", "figure")],
#     Input("rocauc-picker", "value"),
# )
# def update_roc_auc(selected_version):
#     return cal.roc[selected_version]


# @app.callback(Output("confusion-matrix", "figure"), Input("cmx-type", "value"))
# def update_cmx(cmx_type):
#     v = list(cal.mega.keys())[-1]
#     return cal.triple_cmx(cal.cmx["counts"], cmx_type, classes=["2GB", "8GB", "16GB", "64GB"])
