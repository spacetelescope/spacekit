from dash import dcc
from dash import html
from spacekit.dashboard.cal.config import hst

layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Br(),
                dcc.Link("Home", href="/"),
                html.Br(),
                dcc.Link("Evaluation", href="/eval"),
                html.Br(),
                dcc.Link("Prediction", href="/pred"),
                html.Br(),
            ]
        ),
        html.Div(
            id="explore-dataset",
            children=[
                # BOX PLOTS: CONTINUOUS VARS (N_FILES + TOTAL_MB)
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id="box-vars",
                                                    options=[
                                                        {"label": t, "value": t}
                                                        for t in [
                                                            "features",
                                                            "targets",
                                                        ]
                                                    ],
                                                    value="targets",
                                                )
                                            ],
                                            style={
                                                "width": "40%",
                                                "display": "inline-block",
                                                "padding": 5,
                                                "float": "center",
                                                "color": "black",
                                            },
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="boxplot00",
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": "50%",
                                    },
                                ),
                                dcc.Graph(
                                    id="boxplot10",
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": "50%",
                                    },
                                ),
                                dcc.Graph(
                                    id="boxplot01",
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": "50%",
                                    },
                                ),
                                dcc.Graph(
                                    id="boxplot11",
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": "50%",
                                    },
                                ),
                            ],
                            style={"color": "white", "width": "100%"},
                        )
                    ],
                    style={
                        "backgroundColor": "#242a44",
                        "color": "white",
                        "padding": 15,
                        "display": "inline-block",
                        "width": "85%",
                    },
                ),
                html.Div(
                    children=[
                        # FEATURE COMPARISON SCATTERPLOTS
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="xaxis-features",
                                            options=[
                                                {"label": f, "value": f}
                                                for f in hst.feature_list
                                            ],
                                            value="n_files",
                                        )
                                    ],
                                    style={
                                        "width": "20%",
                                        "display": "inline-block",
                                        "padding": 5,
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="yaxis-features",
                                            options=[
                                                {"label": t, "value": t}
                                                for t in ["memory", "wallclock"]
                                            ],
                                            value="memory",
                                        )
                                    ],
                                    style={
                                        "width": "20%",
                                        "display": "inline-block",
                                        "padding": 5,
                                    },
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id="acs-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="wfc3-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="cos-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="stis-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
                ),
            ],
        ),
    ],
    style={
        "backgroundColor": "#242a44",
        "color": "white",
        "padding": 20,
        "display": "inline-block",
        "width": "100%",
        "textAlign": "center",
    },
)
