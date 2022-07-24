from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from argparse import ArgumentParser
from spacekit.dashboard.cal.app import app
from spacekit.dashboard.cal import eda, eval, pred, nodegraph
from spacekit.dashboard.cal.config import cal, hst, tx_file
from spacekit.preprocessor.transform import PowerX


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
                html.Div("for the Hubble Space Telescope's"),
                html.Div("data reprocessing pipeline."),
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
    return cal.keras[selected_version]


# ROC AUC CALLBACK
@app.callback(
    [Output("roc-auc", "figure"), Output("precision-recall-fig", "figure")],
    Input("rocauc-picker", "value"),
)
def update_roc_auc(selected_version):
    return cal.roc[selected_version]


# CMX Callback
@app.callback(Output("confusion-matrix", "figure"), Input("cmx-type", "value"))
def update_cmx(cmx_type):
    return cal.triple_cmx(cal.cmx[cmx_type], cmx_type)


# Page 2 EDA callbacks
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
    return hst.scatter[yaxis_name][xaxis_name]


# Box Plots
@app.callback(
    [
        Output("boxplot00", "figure"),
        Output("boxplot10", "figure"),
        Output("boxplot01", "figure"),
        Output("boxplot11", "figure"),
    ],
    Input("box-vars", "value"),
)
def update_box(box_vars):
    if box_vars == "features":
        boxes = ["n_files", "x_files", "total_mb", "x_size"]
    else:
        boxes = ["memory", "mem_fence", "wallclock", "wall_fence"]
    box_figs = [hst.box[b] for b in boxes]
    return box_figs


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
        x_features, tx_file
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
        x_cols = [
            "n_files",
            "total_mb",
            "drizcorr",
            "pctecorr",
            "crsplit",
            "subarray",
            "detector",
            "dtype",
            "instr",
        ]
        return list(cal.df.loc[selected_ipst][x_cols].values)


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
        Px = PowerX(
            x_features,
            cols=["n_files", "total_mb"],
            ncols=[0, 1],
            tx_file=tx_file,
            rename=None,
        )
        X = Px.Xt
        neurons = nodegraph.calculate_neurons(X)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--env", type=str, choices=["prod", "dev"], default="prod")
    args = parser.parse_args()
    host, port = args.host, args.port
    if args.env == "dev":
        app.run_server(host=host, port=port, debug=True, dev_tools_prune_errors=False)
    else:
        app.run_server(host="0.0.0.0", port=8050)
