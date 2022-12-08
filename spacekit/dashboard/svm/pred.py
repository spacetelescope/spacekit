from dash import dcc
from dash import html

# import dash_daq as daq

layout = html.Div(
    children=[
        # nav
        html.Div(
            children=[
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Home",
                    href="/",
                    style={"padding": 5, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Evaluation",
                    href="/eval",
                    style={"padding": 5, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
                dcc.Link(
                    "Analysis",
                    href="/eda",
                    style={"padding": 10, "display": "inline-block"},
                ),
                html.P("|", style={"display": "inline-block"}),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            children=[
                html.H4("Image Augmentation"),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    [
                                        html.Label(
                                            "VISIT",
                                            style={
                                                "paddingTop": 5,
                                                "paddingRight": 5,
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="image-state",
                                            options=[
                                                {
                                                    "label": "_".join(
                                                        i.split("_")[-4:]
                                                    ),
                                                    "value": i,
                                                }
                                                for i in [
                                                    "hst_11099_04_wfc3_ir_total_ia0m04",
                                                    "hst_12062_eh_wfc3_uvis_total_ibeveh",
                                                    "hst_8992_03_acs_wfc_total_j8cw03",
                                                    "hst_9454_11_acs_hrc_total_j8ff11",
                                                ]
                                            ],
                                            value="hst_11099_04_wfc3_ir_total_ia0m04",
                                            style={
                                                "width": 300,
                                                "marginLeft": 5,
                                                "marginRight": 5,
                                                "marginBottom": 5,
                                                "float": "left",
                                                "color": "black",
                                            },
                                        ),
                                        html.Button(
                                            "PREVIEW",
                                            id="preview-button-state",
                                            n_clicks=0,
                                            style={
                                                "width": 120,
                                                "marginLeft": 10,
                                                "marginTop": 0,
                                                "float": "center",
                                            },
                                        ),
                                        html.Button(
                                            "AUGMENT",
                                            id="augment-button-state",
                                            n_clicks=0,
                                            disabled=True,
                                            style={
                                                "width": 120,
                                                "marginLeft": 10,
                                                "marginTop": 0,
                                                "float": "center",
                                            },
                                        ),
                                    ],
                                )
                            ]
                        )
                    ],
                    style={
                        "color": "black",
                        "width": "80%",
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="original-image",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        dcc.Graph(
                            id="augmented-image",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
                ),
            ],
            style={
                "color": "white",
                "background-color": "#242a44",
                "border": "2px #333 solid",
                "borderRadius": 5,
                "width": "80%",
                "margin": 25,
                "padding": 10,
            },
        ),
        html.Div(
            children=[
                html.H4("Alignment Classifier Game"),
                html.P(
                    "Successful or Compromised? See if you can correctly identify which single visit mosaic images are aligned correctly and which ones are 'suspicious' or 'compromised' (likely to be misaligned). Can you beat the classifier algorithm?"
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    [
                                        html.Label(
                                            "VISIT",
                                            style={
                                                "paddingTop": 5,
                                                "paddingRight": 5,
                                                "float": "left",
                                            },
                                        ),
                                        dcc.Dropdown(
                                            id="dropdown-state",
                                            options=[
                                                {
                                                    "label": "_".join(
                                                        i.split("_")[-4:]
                                                    ),
                                                    "value": i,
                                                }
                                                for i in [
                                                    "hst_11099_04_wfc3_ir_total_ia0m04",
                                                    "hst_12062_eh_wfc3_uvis_total_ibeveh",
                                                    "hst_8992_03_acs_wfc_total_j8cw03",
                                                    "hst_9454_11_acs_hrc_total_j8ff11",
                                                ]
                                            ],
                                            value="hst_11099_04_wfc3_ir_total_ia0m04",
                                            style={
                                                "width": 300,
                                                "marginLeft": 5,
                                                "marginRight": 5,
                                                "marginBottom": 5,
                                                "float": "left",
                                                "color": "black",
                                            },
                                        ),
                                        html.Button(
                                            "PREVIEW",
                                            id="button1-state",
                                            n_clicks=0,
                                            style={
                                                "width": 120,
                                                "marginLeft": 10,
                                                "marginTop": 0,
                                                "float": "center",
                                            },
                                        ),
                                        html.Button(
                                            "AUGMENT",
                                            id="button2-state",
                                            n_clicks=0,
                                            style={
                                                "width": 120,
                                                "marginLeft": 10,
                                                "marginTop": 0,
                                                "float": "center",
                                            },
                                        ),
                                    ],
                                )
                            ]
                        )
                    ],
                    style={
                        "color": "black",
                        "width": "80%",
                        "margin": 15,
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="image1",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        # dcc.Graph(
                        #     id="image2",
                        #     style={"display": "inline-block", "float": "center"},
                        # ),
                    ],
                    style={"color": "white", "width": "100%"},
                ),
            ],
            style={
                "color": "white",
                "width": "80%",
                "background-color": "#242a44",
                "border": "2px #333 solid",
                "borderRadius": 5,
                "margin": 25,
                "marginBottom": 10,
                "padding": 10,
            },
        ),
    ],
    style={
        "width": "100%",
        "height": "100%",
        "background-color": "#1b1f34",
        "color": "white",
        "padding": 5,
    },
)
