# TODO: comparative model analysis and evaluation
from dash import dcc
from dash import html
from spacekit.dashboard.svm.config import svm


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
                # Ensemble Model
                html.Div(
                    children=[
                        html.H4(children="SVM Classifier", style={"padding": 10}),
                        # ACCURACY vs LOSS (BARPLOTS)
                        "Accuracy vs Loss",
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="acc-bars",
                                    figure=svm.acc_fig,
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="loss-bars",
                                    figure=svm.loss_fig,
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
                                                        for v in svm.versions
                                                    ],
                                                    value=svm.versions[-1],
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
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="keras-loss",
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
                                                        for v in svm.versions
                                                    ],
                                                    value=svm.versions[-1],
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
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="precision-recall-fig",
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
                                            value="normalized",
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


# # Page 1 EVAL callbacks
# # KERAS CALLBACK
# @app.callback(
#     [Output("keras-acc", "figure"), Output("keras-loss", "figure")],
#     Input("version-picker", "value"),
# )
# def update_keras(selected_version):
#     com = svm.mega[selected_version]["res"]["test"]
#     com.acc_fig = com.keras_acc_plot()
#     com.loss_fig = com.keras_loss_plot()
#     keras_figs = [com.acc_fig, com.loss_fig]
#     return keras_figs


# # ROC AUC CALLBACK
# @app.callback(
#     [Output("roc-auc", "figure"), Output("precision-recall-fig", "figure")],
#     Input("rocauc-picker", "value"),
# )
# def update_roc_auc(selected_version):
#     com = svm.mega[selected_version]["res"]["test"]
#     com.roc_fig = com.make_roc_curve()
#     com.pr_fig = com.make_pr_curve()
#     return [com.roc_fig, com.pr_fig]


# @app.callback(Output("confusion-matrix", "figure"), Input("cmx-type", "value"))
# def update_cmx(cmx_type):
#     # com.cm_fig
#     v = list(svm.mega.keys())[-1]
#     com = svm.mega[v]["res"]["test"]
#     cmx_fig = com.make_cmx_figure(com, cmx_type)
#     return cmx_fig
