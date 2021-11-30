from dash_html_components.P import P
import numpy as np
import itertools

from .predictor import get_model, softmax_activation


def input_layer_weights():
    neurons = clf.layers[1].weights[0].shape[0]
    input_weights = {}
    for i in list(range(neurons)):
        key = f"x{i}"
        input_weights[key] = np.array(clf.layers[1].weights[0][i])
    return input_weights


def dense_layer_weights(lyr, src):
    if lyr == "y":
        layer_num = 7
    else:
        layer_num = int(lyr[1:])
    src_key = src.split("-")[0]
    neurons = clf.layers[layer_num].weights[0].shape[0]  # 18
    dense_weights = {}
    for i in list(range(neurons)):
        key = f"{src_key}-{i}"
        dense_weights[key] = np.array(clf.layers[layer_num].weights[0][i])
    return dense_weights


def make_permutations(src, trg):
    if trg == "y":
        layer_num = 7
    else:
        layer_num = int(trg[1])
    if src != "x":
        src_key = f"{src}-"
    else:
        src_key = "x"
    n1 = clf.layers[layer_num].weights[0].shape[0]
    n2 = clf.layers[layer_num].weights[0].shape[1]
    src_neurons, trg_neurons = [], []
    permutations = []
    for s in list(range(n1)):
        src_neurons.append(f"{src_key}{s}")
    for t in list(range(n2)):
        trg_neurons.append(f"{trg}-{t}")
    for r in itertools.product(src_neurons, trg_neurons):
        permutations.append((r[0], r[1]))
    return permutations


def make_weights():
    x_weights = input_layer_weights()
    # h_src = ['h2', 'h3', 'h4', 'h5', 'h6', 'y']
    # h_trg = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    h1_weights = dense_layer_weights("h2", "h1")
    h2_weights = dense_layer_weights("h3", "h2")
    h3_weights = dense_layer_weights("h4", "h3")
    h4_weights = dense_layer_weights("h5", "h4")
    h5_weights = dense_layer_weights("h6", "h5")
    h6_weights = dense_layer_weights("y", "h6")
    weight_groups = [
        x_weights,
        h1_weights,
        h2_weights,
        h3_weights,
        h4_weights,
        h5_weights,
        h6_weights,
    ]
    weights = []
    for group in weight_groups:
        for arr in group.values():
            for w in arr:
                weights.append(w)
    return weights


def make_edges(weights):
    xh1 = make_permutations("x", "h1")
    h1h2 = make_permutations("h1", "h2")
    h2h3 = make_permutations("h2", "h3")
    h3h4 = make_permutations("h3", "h4")
    h4h5 = make_permutations("h4", "h5")
    h5h6 = make_permutations("h5", "h6")
    h6y = make_permutations("h6", "y")
    permutations = [xh1, h1h2, h2h3, h3h4, h4h5, h5h6, h6y]
    edge_pairs = []
    for p in permutations:
        for index, (source, target) in enumerate(p):
            edge_pairs.append((source, target, weights[index]))
    return edge_pairs


def make_parent_nodes():
    ids = [
        "inputs",
        "dense1",
        "dense2",
        "dense3",
        "dense4",
        "dense5",
        "dense6",
        "outputs",
    ]
    labels = [
        "InputLayer",
        "Dense1",
        "Dense2",
        "Dense3",
        "Dense4",
        "Dense5",
        "Dense6",
        "Outputs",
    ]
    classes = [
        "inputLayer",
        "hiddenLayer",
        "hiddenLayer",
        "hiddenLayer",
        "hiddenLayer",
        "hiddenLayer",
        "hiddenLayer",
        "outputLayer",
    ]
    parent_nodes = list(zip(ids, labels, classes))
    return parent_nodes


def set_origin_points(layer_name):
    xy_origin = {
        "x": (0, 300),
        "h1": (1000, 250),
        "h2": (2500, 150),
        "h3": (5000, 50),
        "h4": (7500, 150),
        "h5": (9000, 250),
        "h6": (10000, 300),
        "y": (11000, 400),
    }
    index = list(enumerate(xy_origin.keys()))
    for i in index:
        if layer_name in i:
            layer_idx = i[0]
    return xy_origin[layer_name], layer_idx


def get_coords(xy_origin, layer_idx):
    x0 = xy_origin[0]
    y0 = xy_origin[1]

    if layer_idx == 0:
        neurons = clf.layers[layer_idx].output_shape[0][1]
    else:
        neurons = clf.layers[layer_idx].units
    slope = int(3200 / neurons)
    xy_coords = []
    for _ in list(range(neurons)):
        x = x0
        y = y0
        xy_coords.append((x, y))
        y0 += slope
    return xy_coords


def make_nodes(layer, neurons, parent):
    ids = []
    parents = []
    for n in list(range(neurons)):
        if layer == "x":
            i = f"{layer}{n}"
        else:
            i = f"{layer}-{n}"
        ids.append(i)
        parents.append(parent)
    if parent == "inputs":
        labels = [
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
    elif parent == "outputs":
        labels = ["2G", "8G", "16G", "64G"]
    else:
        labels = []
        for n in list(range(neurons)):
            labels.append(str(n + 1))

    xy_origin, layer_idx = set_origin_points(layer)
    xy_coords = get_coords(xy_origin, layer_idx)
    xs = [x for (x, _) in xy_coords]
    ys = [y for (_, y) in xy_coords]

    nodes = list(zip(ids, labels, parents, xs, ys))
    return nodes


def make_node_groups():
    layers = ["x", "h1", "h2", "h3", "h4", "h5", "h6", "y"]
    neurons = [9, 18, 32, 64, 32, 18, 9, 4]
    parents = [
        "inputs",
        "dense1",
        "dense2",
        "dense3",
        "dense4",
        "dense5",
        "dense6",
        "outputs",
    ]
    node_groups = []
    for l, n, p in zip(layers, neurons, parents):
        group = make_nodes(l, n, p)
        for id, label, parent, x, y in group:
            node_groups.append((id, label, parent, x, y))
    return node_groups


def edge_weight_clicks(src, trg):
    input_weights = input_layer_weights()  # delete
    lyr, idx = trg.split("-")
    if src[0] == "x":
        w = input_weights[src][int(idx)]
    elif src[0] == "h":
        weights = dense_layer_weights(lyr, src)
        w = weights[src][int(idx)]
    else:
        return None
    return w


def node_bias_clicks(node):
    node_layer = node.split("-")
    if len(node_layer) == 1:
        bias = None
    else:
        lyr, idx = node.split("-")  # h1, 0
        if lyr == "y":
            layer_num = 7
        else:
            layer_num = int(lyr[1:])
        bias = float(np.array(clf.layers[layer_num].bias[int(idx)]))
    return bias


def nodes_edges(parent_nodes, node_groups, edge_pairs):
    nodes = [
        {"data": {"id": id, "label": label}, "classes": layerclass}
        for id, label, layerclass in parent_nodes
    ]
    nodes.extend(
        [
            {
                "data": {"id": id, "label": label, "parent": parent},
                "position": {"x": x, "y": y},
            }
            for id, label, parent, x, y in node_groups
        ]
    )
    edges = [
        {"data": {"source": source, "target": target, "weight": weight}}
        for source, target, weight in edge_pairs
    ]
    return nodes, edges


def initialize_nodes(parent_nodes, node_groups):
    # network layers (rectangles containing each group of neurons)
    nodes = [
        {"data": {"id": id, "label": label}, "classes": layerclass}
        for id, label, layerclass in parent_nodes
    ]
    # add input layer nodes (no classes)
    nodes.extend(
        [
            {
                "data": {"id": id, "label": label, "parent": parent},
                "position": {"x": x, "y": y},
            }
            for id, label, parent, x, y in node_groups[:9]
        ]
    )
    return nodes


def update_node_groups(node_groups, neurons):
    # determine neuron classes: 'activated' if n>0, else ''
    N = []
    for layer in neurons:
        N.extend([n for n in layer])
    num_dense = len(N) - len(neurons[-1])
    node_groups_updated = []
    inactive = []
    for idx, (id, label, parent, x, y) in enumerate(node_groups[9:]):
        v = N[idx]
        if idx < num_dense:
            if v == 0:
                classes = "activatedNode"
            else:
                inactive.append(id)
                classes = ""
        else:
            if v == np.max(N[-4:]):
                classes = "yPred"
            else:
                classes = ""
        node_groups_updated.append((id, label, parent, x, y, classes))
    return inactive, node_groups_updated


def update_edge_pairs(inactive, edge_pairs):
    # determine weight classes: 'deadRelu' if n==0, else ''
    edge_pairs_updated = []
    for source, target, weight in edge_pairs:
        if source in inactive:
            classes = "inactiveRelu"
        elif target in inactive:
            classes = "inactiveRelu"
        else:
            classes = ""
        edge_pairs_updated.append((source, target, weight, classes))
    return edge_pairs_updated


def activate_neurons(parent_nodes, node_groups, edge_pairs, neurons):
    inactive, node_groups_updated = update_node_groups(node_groups, neurons)
    edge_pairs_updated = update_edge_pairs(inactive, edge_pairs)

    nodes = initialize_nodes(parent_nodes, node_groups)
    # update nodes list
    nodes.extend(
        [
            {
                "data": {"id": id, "label": label, "parent": parent},
                "classes": classes,
                "position": {"x": x, "y": y},
            }
            for id, label, parent, x, y, classes in node_groups_updated
        ]
    )
    edges = [
        {
            "data": {"source": source, "target": target, "weight": weight},
            "classes": classes,
        }
        for source, target, weight, classes in edge_pairs_updated
    ]
    return nodes, edges


def make_styles():
    styles = {
        "pre": {
            #'border': 'thin lightgrey solid',
            "overflowX": "scroll",
            "display": "inline-block",
            "float": "left",
            "width": 400,
            "background-color": "#1b1f34",
            "margin": 0,
            "padding": 0,
            "text-align": "center",
        },
        "gradbar-blue": {
            "color": "black",
            "display": "inline-block",
            "float": "left",
            "padding": 5,
            "background-color": "linear-gradient(145deg, rgba(33, 134, 244, 0.5) 0%, rgba(33, 134, 244, 0.4) 100%)",
            "background-image": "rgb(0, 0, 0)",
            "background-blend-mode": "overlay",
            "box-shadow": "rgb(0 0 0 / 45%) 2px 2px 6px 1px, rgb(255 255 255 / 30%) 1px 1px 2px 0px inset, rgb(0 0 0 / 60%) 1px 1px 1px 0px, rgb(33 134 244) 0px 0px 3px 0px",
        },
        "gradbar-green": {
            "color": "black",
            "display": "inline-block",
            "float": "left",
            "padding": 5,
            #'background-color': '#242a44',
            "background-color": "linear-gradient(145deg, rgba(0, 234, 100, 0.5) 0%, rgba(0, 234, 100, 0.4) 100%)",
            "background-image": "rgb(0, 0, 0)",
            #'background-image': 'linear-gradient(145deg, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.4) 100%)',
            "background-blend-mode": "overlay",
            "box-shadow": "rgb(0 0 0 / 45%) 2px 2px 6px 1px, rgb(255 255 255 / 30%) 1px 1px 2px 0px inset, rgb(0 0 0 / 60%) 1px 1px 1px 0px, rgb(0 234 100) 0px 0px 3px 0px",
        },
    }
    return styles


def make_stylesheet():
    stylesheet = [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "text-halign": "center",
            },
        },
        {
            "selector": ".inputLayer",
            "style": {"background-color": "cyan", "color": "white"},
        },
        {
            "selector": ".outputLayer",
            "style": {"background-color": "hotpink", "color": "white"},
        },
        {
            "selector": ".hiddenLayer",
            "style": {"background-color": "#00EA64", "color": "white"},
        },
        {
            "selector": ".activatedNode",
            "style": {
                "background-color": "black",
                "opacity": 1,
                "border-color": "black",
                "border-width": 5,
                "border-opacity": 1,
            },
        },
        {
            "selector": ".yPred",
            "style": {
                "background-color": "black",
                "border-color": "black",
                "border-width": 5,
                "border-opacity": 1,
                "opacity": 1,
            },
        },
        {
            "selector": ".inactiveRelu",
            "style": {"line-color": "transparent", "opacity": 0.05, "z-index": -1},
        },
    ]
    return stylesheet


def make_neural_graph(NN=None, neurons=None):
    if NN is None:
        global clf
        clf = get_model("./models/mem_clf")
    weights = make_weights()
    edge_pairs = make_edges(weights)
    parent_nodes = make_parent_nodes()
    node_groups = make_node_groups()
    if neurons is None:
        nodes, edges = nodes_edges(parent_nodes, node_groups, edge_pairs)
    else:
        nodes, edges = activate_neurons(parent_nodes, node_groups, edge_pairs, neurons)
    return nodes, edges
