# import flask
import glob
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_daq as daq
from dash.exceptions import PreventUpdate

from spacekit.analyzer.explore import HSTRepro
from spacekit.analyzer.compute import Computer
from spacekit.extractor import load_data
from spacekit.dashboard.pred import predictor, nodegraph

app = dash.Dash(__name__, title="spacekit dashboard", suppress_callback_exceptions=True)

global dataset
global clf
datasets =  sorted(list(glob.glob(f"data/20??-*-*-*")))
print(datasets)
# timestamps = [1636048291, 1635457222, 1629663047]
# versions = ["2021-11-04", "2021-10-28", "2021-08-22"]

timestamps = [int(t.split('-')[-1]) for t in datasets]
versions = [v[5:15] for v in datasets]

selection = f"{versions[-1]}-{timestamps[-1]}"

dataset = f"data/{selection}/latest.csv"
print(dataset)

meta = load_data.make_meta(timestamps, versions)
results = load_data.make_res(meta, versions)
#df_scores = load_data.get_scores(results)
# df_meta = load_data.import_csv(src='file', key='./data/training_metadata.csv')

# LOAD DATA AND FIGURES

com = Computer("mem_clf", "test", [0,1,2,3], show=False)
com.res_path = f"data/{selection}/results"
com.upload()
# LOAD TRAINING DATASET (for scatterplot)
df = load_data.get_single_dataset(dataset)
hst = HSTRepro(df)
hst.df_by_instr()


model_path = f"data/{selection}/models"
clf = predictor.get_model(f"{model_path}/mem_clf")

# NN = {
#     "clf": f"{model_path}/mem_clf/", 
#     "mem_reg": f"{model_path}/mem_reg/",
#     "wall_reg": f"{model_path}/wall_reg/"
#     }

# versions = ["v0", "v1", "v2"]

stylesheet = nodegraph.make_stylesheet()
styles = nodegraph.make_styles()
edges, nodes = nodegraph.make_neural_graph()

url_bar_and_content_div = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

layout_index = html.Div(
    children=[
        html.Br(),
        html.H1("SPACEKIT", style={"padding": 15}),
        html.H2("Dashboard"),
        html.Div(
            children=[
                html.Div("Model Performance + Statistical Analysis"),
                html.Div("for the Hubble Space Telescope's"),
                html.Div("data reprocessing pipeline."),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Br(),
                dcc.Link("Evaluation", href="/page-1"),
                html.Br(),
                dcc.Link("Analysis", href="/page-2"),
                html.Br(),
                dcc.Link("Prediction", href="/page-3"),
            ]
            )
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


layout_page_1 = html.Div(
    children=[
        html.Div(
            children=[
                html.Br(),
                dcc.Link("Home", href="/"),
                html.Br(),
                dcc.Link("Analysis", href="/page-2"),
                html.Br(),
                dcc.Link("Prediction", href="/page-3"),
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
                                    figure=com.acc_fig,
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "padding": 25,
                                    },
                                ),
                                dcc.Graph(
                                    id="loss-bars",
                                    figure=com.loss_fig,
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
                                                    id="version-picker",
                                                    options=[
                                                        {"label": str(v), "value": v}
                                                        for v in versions
                                                    ],
                                                    value=versions[-1],
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
                                                        for v in versions
                                                    ],
                                                    value=versions[-1],
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


layout_page_2 = html.Div(
    children=[
        html.Div(
            children=[
                html.Br(),
                dcc.Link("Home", href="/"),
                html.Br(),
                dcc.Link("Evaluation", href="/page-1"),
                html.Br(),
                dcc.Link("Prediction", href="/page-3"),
                html.Br(),
            ]
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


layout_page_3 = html.Div(
    children=[
        # nav
        html.Div(
            children=[
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Home", href="/", style={"padding": 5, "display": "inline-block"}
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Evaluation",
                    href="/page-1",
                    style={"padding": 5, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Analysis",
                    href="/page-2",
                    style={"padding": 10, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
            ],
            style={"display": "inline-block"},
        ),
        # GRAPH
        html.Div(
            children=[
                cyto.Cytoscape(
                    id="cytoscape-compound",
                    layout={"name": "preset"},
                    style={
                        "width": "99vw",
                        "height": "60vh",
                        "display": "inline-block",
                        "float": "center",
                        "background-color": "#1b1f34",
                    },
                    stylesheet=stylesheet,
                    elements=edges + nodes,
                ),
                html.Div(
                    children=[
                        html.P(id="cytoscape-tapNodeData-output", style=styles["pre"]),
                        html.P(id="cytoscape-tapEdgeData-output", style=styles["pre"]),
                        html.P(
                            id="cytoscape-mouseoverNodeData-output", style=styles["pre"]
                        ),
                        html.P(
                            id="cytoscape-mouseoverEdgeData-output", style=styles["pre"]
                        ),
                    ],
                    style={
                        "width": "100%",
                        "margin": 0,
                        "padding": 0,
                        "display": "inline-block",
                        "float": "left",
                        "background-color": "#1b1f34",
                    },
                ),
            ]
        ),
        # CONTROLS
        html.Div(
            id="controls",
            children=[
                # # INPUT DROPDOWNS
                html.Div(
                    id="x-features",
                    children=[
                        # FEATURE SELECTION DROPDOWNS (Left)
                        html.Div(
                            id="inputs-one",
                            children=[
                                html.Label(
                                    [
                                        html.Label(
                                            "IPPPSSOOT",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="select-ipst",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in df.index.values
                                            ],
                                            value="idio03010",
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "INSTR",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="instr-state",
                                            options=[
                                                {"label": "ACS", "value": 0},
                                                {"label": "COS", "value": 1},
                                                {"label": "STIS", "value": 2},
                                                {"label": "WFC3", "value": 3},
                                            ],
                                            value=0,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "DTYPE",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="dtype-state",
                                            options=[
                                                {"label": "SINGLETON", "value": 0},
                                                {"label": "ASSOCIATION", "value": 1},
                                            ],
                                            value=0,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "DETECTOR",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="detector-state",
                                            options=[
                                                {"label": "IR/HRC/SBC", "value": 0},
                                                {"label": "UVIS/WFC", "value": 1},
                                            ],
                                            value=0,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "SUBARRAY",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="subarray-state",
                                            options=[
                                                {"label": "TRUE", "value": 1},
                                                {"label": "FALSE", "value": 0},
                                            ],
                                            value=0,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                )
                                # END outputs Left Col
                            ],
                            style={
                                "display": "inline-block",
                                "float": "left",
                                "padding": 5,
                                "width": 270,
                            },
                        ),  #'border': 'thin lightgrey solid',
                        # INPUTS RIGHT COL
                        html.Div(
                            id="inputs-two",
                            children=[
                                html.Label(
                                    [
                                        html.Label(
                                            "PCTECORR",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="pctecorr-state",
                                            options=[
                                                {"label": "OMIT", "value": 0},
                                                {"label": "PERFORM", "value": 1},
                                            ],
                                            value=1,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "DRIZCORR",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="drizcorr-state",
                                            options=[
                                                {"label": "OMIT", "value": 0},
                                                {"label": "PERFORM", "value": 1},
                                            ],
                                            value=1,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "CRSPLIT",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        daq.NumericInput(
                                            id="crsplit-state",
                                            value=2,
                                            min=0,
                                            max=2,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "TOTAL_MB",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        daq.NumericInput(
                                            id="totalmb-state",
                                            value=4,
                                            min=0,
                                            max=900,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                ),
                                html.Label(
                                    [
                                        html.Label(
                                            "N_FILES",
                                            style={
                                                "padding": 5,
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        ),
                                        daq.NumericInput(
                                            id="nfiles-state",
                                            value=2,
                                            min=1,
                                            max=200,
                                            style={
                                                "color": "black",
                                                "width": 135,
                                                "display": "inline-block",
                                                "float": "right",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 255,
                                    },
                                )
                                # END Input Right COL
                            ],
                            style={
                                "display": "inline-block",
                                "float": "left",
                                "padding": 5,
                                "width": 270,
                            },
                        ),  #'border': 'thin lightgrey solid',
                        # END FEATURE INPUTS
                    ],
                    style={
                        "display": "inline-block",
                        "float": "left",
                        "paddingTop": 20,
                        "paddingBottom": 5,
                        "paddingLeft": "2.5%",
                        "paddingRight": "2.5%",
                        "background-color": "#242a44",
                        "min-height": 311,
                    },
                ),  #'border': 'thin lightgreen solid',
                # MEMORY PRED VS ACTUAL
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Button(
                                            "PREDICT",
                                            id="submit-button-state",
                                            n_clicks=0,
                                            style={"width": 110},
                                        )
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": 120,
                                        "paddingTop": 15,
                                        "paddingLeft": 15,
                                    },
                                ),
                                html.Div(
                                    [
                                        # Memory Bin Prediction LED Display
                                        daq.LEDDisplay(
                                            id="prediction-bin-output",
                                            # label="PRED",
                                            # labelPosition='bottom',
                                            value="0",
                                            color="#2186f4",
                                            size=64,
                                            backgroundColor="#242a44",
                                            # style={'display': 'inline-block', 'float': 'center'}
                                        )
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "paddingTop": 20,
                                        "paddingBottom": 5,
                                        "paddingLeft": 30,
                                        "width": 120,
                                    },
                                ),
                                html.Div(
                                    [
                                        daq.BooleanSwitch(
                                            id="activate-button-state",
                                            on=False,
                                            label="ACTIVATE",
                                            labelPosition="bottom",
                                            color="#2186f4",
                                        )  # '#00EA64'
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "center",
                                        "width": 120,
                                        "paddingTop": 15,
                                        "paddingLeft": 10,
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "float": "left",
                                "padding": 5,
                                "width": 140,
                            },
                        ),  #'border': 'thin lightgrey solid',
                        # Probabilities
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        daq.GraduatedBar(
                                            id="p0",
                                            label="P(0)",
                                            labelPosition="bottom",
                                            step=0.05,
                                            min=0,
                                            max=1,
                                            value=1.0,
                                            showCurrentValue=True,
                                            vertical=True,
                                            size=220,
                                            color="rgb(33, 134, 244)",
                                            style=styles["gradbar-blue"],
                                        ),
                                        daq.GraduatedBar(
                                            id="p1",
                                            label="P(1)",
                                            labelPosition="bottom",
                                            step=0.05,
                                            min=0,
                                            max=1,
                                            value=0.30,
                                            showCurrentValue=True,
                                            vertical=True,
                                            size=220,
                                            color="rgb(33, 134, 244)",
                                            style=styles["gradbar-blue"],
                                        ),
                                        daq.GraduatedBar(
                                            id="p2",
                                            label="P(2)",
                                            labelPosition="bottom",
                                            step=0.05,
                                            min=0,
                                            max=1,
                                            value=0.60,
                                            showCurrentValue=True,
                                            vertical=True,
                                            size=220,
                                            color="rgb(33, 134, 244)",
                                            style=styles["gradbar-blue"],
                                        ),
                                        daq.GraduatedBar(
                                            id="p3",
                                            label="P(3)",
                                            labelPosition="bottom",
                                            step=0.05,
                                            min=0,
                                            max=1,
                                            value=0.10,
                                            showCurrentValue=True,
                                            vertical=True,
                                            size=220,
                                            color="rgb(33, 134, 244)",
                                            style=styles["gradbar-blue"],
                                        )
                                        # END Probabilities
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 175,
                                    },
                                )
                            ],
                            style={
                                "display": "inline-block",
                                "float": "left",
                                "padding": 5,
                                "width": 190,
                            },
                        ),  #'border': 'thin lightgrey solid',
                    ],
                    style={
                        "display": "inline-block",
                        "float": "left",
                        "paddingTop": 5,
                        "paddingBottom": 5,
                        "paddingLeft": "2.5%",
                        "paddingRight": "2.5%",
                        "background-color": "#242a44",
                        "min-height": 326,
                    },
                ),  #'border': 'thin lightgreen solid',
                # Memory GAUGE Predicted vs Actual
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        daq.Gauge(
                                            id="memory-gauge-predicted",
                                            color="#2186f4",  #'#00EA64',
                                            label="Memory (pred)",
                                            labelPosition="bottom",
                                            units="GB",
                                            showCurrentValue=True,
                                            max=64,
                                            min=0,
                                            size=175,
                                            style={
                                                "color": "white",
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        )
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 200,
                                    },
                                ),
                                html.Div(
                                    children=[
                                        daq.Gauge(
                                            id="wallclock-gauge-predicted",
                                            color="#2186f4",  #'#00EA64',
                                            value=4500,
                                            label="Wallclock (pred)",
                                            labelPosition="bottom",
                                            units="SECONDS",
                                            showCurrentValue=True,
                                            max=72000,
                                            min=0,
                                            size=175,
                                            style={
                                                "color": "white",
                                                "display": "inline-block",
                                                "float": "left",
                                            },
                                        )
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "float": "left",
                                        "margin": 5,
                                        "width": 200,
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "float": "left",
                                "padding": 5,
                                "width": 440,
                            },
                        )  #'border': 'thin lightgrey solid',
                    ],
                    style={
                        "display": "inline-block",
                        "float": "left",
                        "paddingTop": 5,
                        "paddingBottom": 5,
                        "paddingLeft": "2.5%",
                        "paddingRight": "2.5%",
                        "background-color": "#242a44",
                        "min-height": 326,
                    },
                ),  #'border': 'thin lightgreen solid',
                # END Controls and Outputs
            ],
            style={
                "width": "100%",
                "display": "inline-block",
                "float": "center",
                "background-color": "#242a44",
            },
        )
        # PAGE LAYOUT
    ],
    style={
        "width": "100%",
        "height": "100%",
        "background-color": "#242a44",
        "color": "white",
    },
)


# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div(
    [url_bar_and_content_div, layout_index, layout_page_1, layout_page_2, layout_page_3]
)

# Index callbacks
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    elif pathname == "/page-3":
        return layout_page_3
    else:
        return layout_index




# Page 1 callbacks
# KERAS CALLBACK
@app.callback(
    [Output("keras-acc", "figure"), Output("keras-loss", "figure")],
    Input("version-picker", "value"),
)
def update_keras(selected_version):
    history = results[selected_version]["mem_bin"]["history"]
    keras_figs = scoring.keras_plots(history)
    return keras_figs


# ROC AUC CALLBACK
@app.callback(
    [Output("roc-auc", "figure"), Output("precision-recall-fig", "figure")],
    Input("rocauc-picker", "value"),
)
def update_roc_auc(selected_version):
    y = results[selected_version]["mem_bin"]["y_true"]
    y_scores = results[selected_version]["mem_bin"]["proba"]
    
    return [com.roc_fig, com.pr_fig]


@app.callback(Output("confusion-matrix", "figure"), Input("cmx-type", "value"))
def update_cmx(cmx_type):
    #com.cm_fig
    cmx_fig = com.make_cmx_figure(results[-3:-1], cmx_type)
    return cmx_fig


# Page 2 callbacks
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


# PAGE 3 CALLBACKS


@app.callback(
    [
        Output("prediction-bin-output", "value"),
        Output("memory-gauge-predicted", "value"),
        Output("wallclock-gauge-predicted", "value"),
        Output("p0", "value"),
        Output("p1", "value"),
        Output("p2", "value"),
        Output("p3", "value"),
        Output("activate-button-state", "on"),
    ],
    Input("submit-button-state", "n_clicks"),
    State("nfiles-state", "value"),
    State("totalmb-state", "value"),
    State("drizcorr-state", "value"),
    State("pctecorr-state", "value"),
    State("crsplit-state", "value"),
    State("subarray-state", "value"),
    State("detector-state", "value"),
    State("dtype-state", "value"),
    State("instr-state", "value"),
)
def update_xi_predictions(
    n_clicks,
    n_files,
    total_mb,
    drizcorr,
    pctecorr,
    crsplit,
    subarray,
    detector,
    dtype,
    instr,
):
    if n_clicks == 0:
        raise PreventUpdate
    # if n_clicks > 0:
    x_features = predictor.read_inputs(
        n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr
    )
    m_preds = predictor.make_preds(
        x_features
    )  # [membin, memval, clocktime, p0, p1, p2, p3]
    n_clicks = 0
    m_preds.append(0)  # reset `activate` toggle switch = off
    return m_preds


@app.callback(
    [
        Output("nfiles-state", "value"),
        Output("totalmb-state", "value"),
        Output("drizcorr-state", "value"),
        Output("pctecorr-state", "value"),
        Output("crsplit-state", "value"),
        Output("subarray-state", "value"),
        Output("detector-state", "value"),
        Output("dtype-state", "value"),
        Output("instr-state", "value"),
    ],
    Input("select-ipst", "value"),
)
def update_ipst(selected_ipst):
    if selected_ipst:
        inputs = {
            "n_files": 0,
            "total_mb": 0,
            "drizcorr": 0,
            "pctecorr": 0,
            "crsplit": 0,
            "subarray": 0,
            "detector": 0,
            "dtype": 0,
            "instr": 0,
        }
        data = df.loc[selected_ipst]
        for key in list(inputs.keys()):
            inputs[key] = data[key]

        return list(inputs.values())


#
@app.callback(
    Output("cytoscape-compound", "elements"),
    Input("activate-button-state", "on"),
    State("nfiles-state", "value"),
    State("totalmb-state", "value"),
    State("drizcorr-state", "value"),
    State("pctecorr-state", "value"),
    State("crsplit-state", "value"),
    State("subarray-state", "value"),
    State("detector-state", "value"),
    State("dtype-state", "value"),
    State("instr-state", "value"),
)
def activate_network(
    on, n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr
):
    if on == True:
        x_features = predictor.read_inputs(
            n_files,
            total_mb,
            drizcorr,
            pctecorr,
            crsplit,
            subarray,
            detector,
            dtype,
            instr,
        )
        prep = predictor.Preprocess(x_features)
        prep.inputs = prep.scrub_keys()
        prep.load_pt_data()
        X = prep.transformer()
        neurons = predictor.calculate_neurons(X)
        edges, nodes = nodegraph.make_neural_graph(neurons=neurons)
    else:
        edges, nodes = nodegraph.make_neural_graph(neurons=None)
    return edges + nodes


@app.callback(
    Output("cytoscape-tapNodeData-output", "children"),
    Input("cytoscape-compound", "tapNodeData"),
)
def displayTapNodeData(data):
    if data:
        node = data["id"]
        if node[0] not in ["x", "i"]:
            b = nodegraph.node_bias_clicks(node)
        else:
            b = None
        return f"bias: {node} = {str(b)}"


@app.callback(
    Output("cytoscape-tapEdgeData-output", "children"),
    Input("cytoscape-compound", "tapEdgeData"),
)
def displayTapEdgeData(data):
    if data:
        src = data["source"]  # x1
        trg = data["target"]  # h1-1
        w = nodegraph.edge_weight_clicks(src, trg)
        return f"weight: {src} and {trg} = {str(w)}"


@app.callback(
    Output("cytoscape-mouseoverNodeData-output", "children"),
    Input("cytoscape-compound", "mouseoverNodeData"),
)
def displayTapNodeData(data):
    if data:
        node = data["id"]
        if node[0] not in ["x", "i"]:
            b = nodegraph.node_bias_clicks(node)
        else:
            b = None
        return f"bias: {node} = {str(b)}"


@app.callback(
    Output("cytoscape-mouseoverEdgeData-output", "children"),
    Input("cytoscape-compound", "mouseoverEdgeData"),
)
def displayTapEdgeData(data):
    if data:
        src = data["source"]  # x1
        trg = data["target"]  # h1-1
        w = nodegraph.edge_weight_clicks(src, trg)
        return f"weight: {src} and {trg} = {str(w)}"


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)