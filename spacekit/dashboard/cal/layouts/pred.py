from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import dash_daq as daq
from dash.exceptions import PreventUpdate

from spacekit.dashboard.cal import nodegraph
from spacekit.dashboard.cal.app import app
from spacekit.dashboard.cal.config import NN, tx_file, df
from spacekit.preprocessor.transform import CalX

# global clf
# clf = nodegraph.get_model(f"{model_path}/mem_clf")

# global mem_reg
# mem_reg = nodegraph.get_model(f"{model_path}/mem_reg")

# global wall_reg
# wall_reg = nodegraph.get_model(f"{model_path}/wall_reg")

# global tx_file
# tx_file = f"{model_path}/pt_transform"


stylesheet = nodegraph.make_stylesheet()
styles = nodegraph.make_styles()
edges, nodes = nodegraph.make_neural_graph(NN=NN)

layout = html.Div(
    children=[
        # nav
        html.Div(
            children=[
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Home",
                    href="/layouts/home",
                    style={"padding": 5, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Evaluation",
                    href="/layouts/eval",
                    style={"padding": 5, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Analysis",
                    href="/layouts/eda",
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
                        ),  # 'border': 'thin lightgrey solid',
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
                        ),  # 'border': 'thin lightgrey solid',
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
                ),  # 'border': 'thin lightgreen solid',
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
                        ),  # 'border': 'thin lightgrey solid',
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
                        ),  # 'border': 'thin lightgrey solid',
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
                ),  # 'border': 'thin lightgreen solid',
                # Memory GAUGE Predicted vs Actual
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        daq.Gauge(
                                            id="memory-gauge-predicted",
                                            color="#2186f4",  # '#00EA64',
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
                                            color="#2186f4",  # '#00EA64',
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
                        )  # 'border': 'thin lightgrey solid',
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
                ),  # 'border': 'thin lightgreen solid',
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
    x_features = nodegraph.read_inputs(
        n_files, total_mb, drizcorr, pctecorr, crsplit, subarray, detector, dtype, instr
    )
    m_preds = nodegraph.make_preds(
        x_features, tx_file, NN=NN
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
    if on is True:
        x_features = nodegraph.read_inputs(
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
        modX = CalX(x_features, tx_file)
        neurons = nodegraph.calculate_neurons(modX.X)
        edges, nodes = nodegraph.make_neural_graph(model=NN["mem_clf"], neurons=neurons)
    else:
        edges, nodes = nodegraph.make_neural_graph(model=NN["mem_clf"], neurons=None)
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
def displayMouseNodeData(data):
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
def displayMouseEdgeData(data):
    if data:
        src = data["source"]  # x1
        trg = data["target"]  # h1-1
        w = nodegraph.edge_weight_clicks(src, trg)
        return f"weight: {src} and {trg} = {str(w)}"
