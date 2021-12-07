
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from spacekit.analyzer.explore import HstCalPlots
from spacekit.extractor.load_data import import_dataset
from spacekit.dashboard.cal.app import app
from spacekit.dashboard.cal.config import scanner, df, hst

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
        html.Div(
            id="explore-dataset",
            children=[
                html.Div(
                    [
                    dcc.Dropdown(
                        id="dataset-selector",
                        options=[
                            {"label": n, "value": d} for (n, d) in list(zip(scanner.datasets, list(range(len(scanner.datasets)))))
                        ],
                        value=scanner.primary
                        )
                    ],
                    style={
                        "display": "inline-block",
                        "padding": 5,
                    }
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
                                        {"label": f, "value": f} for f in hst.feature_list
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
                                        {"label": f, "value": f} for f in hst.feature_list
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
                                            id="continuous-vars",
                                            options=[
                                                {"label": "Raw Data", "value": "raw"},
                                                {
                                                    "label": "Normalized",
                                                    "value": "norm",
                                                },
                                            ],
                                            value="raw",
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
                            id="n_files",
                            style={
                                "display": "inline-block",
                                "float": "center",
                                "width": "50%",
                            },
                        ),
                        dcc.Graph(
                            id="total_mb",
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
            ])
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

# Page 2 EDA callbacks
@app.callback(
    [Output("explore-dataset", "children")],
    [Input("dataset-selector", "value")]
)
def data_explorer(dataset_selection):
    scanner.primary = dataset_selection
    scanner.data = scanner.select_dataset() # "data/2021-11-04-1636048291/latest.csv"
    df = import_dataset(
        filename=scanner.data, kwargs=dict(index_col="ipst"), 
        decoder_key={"instr": {0: "acs", 1: "cos", 2: "stis", 3: "wfc3"}}
        )
    hst = HstCalPlots(df)
    hst.df_by_instr()

# SCATTER CALLBACK
@app.callback(
    [
        Output("acs-scatter", "figure"),
        Output("wfc3-scatter", "figure"),
        Output("cos-scatter", "figure"),
        Output("stis-scatter", "figure"),
    ],
    [Input("xaxis-features", "value"), Input("yaxis-features", "value")],
)
def update_scatter(xaxis_name, yaxis_name):
    scatter_figs = hst.make_scatter_figs(xaxis_name, yaxis_name)
    return scatter_figs


@app.callback(
    [Output("n_files", "figure"), Output("total_mb", "figure")],
    Input("continuous-vars", "value"),
)
def update_continuous(raw_norm):
    if raw_norm == "raw":
        vars = ["n_files", "total_mb"]
    elif raw_norm == "norm":
        vars = ["x_files", "x_size"]
    continuous_figs = hst.make_continuous_figs(vars)
    return continuous_figs
