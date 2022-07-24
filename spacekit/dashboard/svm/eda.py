# TODO: eda for regression test data (MLP data)
from dash import dcc
from dash import html
from spacekit.dashboard.svm.config import hst

layout = html.Div(
    children=[
        html.H3("Exploratory Data Analysis"),
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
        # FEATURE BARPLOTS
        html.Div(
            children=[
                html.H4("Mean Feature Value by Detector"),
                html.Div(
                    children=[
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
                            value="rms_ra",
                        )
                    ],
                    style={
                        "color": "black",
                        "width": "20%",
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="bar-group",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
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
        # KDE
        html.Div(
            children=[
                html.H4("Kernel Density Estimates"),
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id="selected-kde",
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
                            value="rms_ra",
                        )
                    ],
                    style={
                        "color": "black",
                        "width": "20%",
                        "display": "inline-block",
                        "padding": 5,
                    },
                ),
                html.Div(
                    children=[
                        html.H4("Kernel Density Estimate by Target"),
                        dcc.Graph(
                            id="kde-targ",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        html.H4("Normalized KDE"),
                        dcc.Graph(
                            id="kde-norm",
                            style={"display": "inline-block", "float": "center"},
                        ),
                        html.H4("KDE rms_ra vs rms_dec"),
                        dcc.Graph(
                            id="kde-rms",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ]
                ),
            ]
        ),
        # FEATURE SCATTERPLOTS
        html.Div(
            children=[
                html.H4(children=["Feature Scatterplots by Detector"]),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="selected-scatter",
                            options=[
                                {"label": f, "value": f}
                                for f in ["rms_ra_dec", "point_segment"]
                            ],
                            value="rms_ra_dec",
                        )
                    ],
                    style={
                        "color": "black",
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
                            id="wfc-scatter",
                            style={"display": "inline-block", "float": "center"},
                        ),
                    ],
                    style={"color": "white", "width": "100%"},
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
        html.Div(
            children=[
                html.Div(
                    dcc.Graph(
                        id="scatter-3d",
                        figure=hst.scatter3d(
                            "point",
                            "segment",
                            "gaia",
                        ),
                        style={
                            "display": "inline-block",
                            "float": "center",
                            "height": 700,
                        },
                    )
                ),
                html.Div(
                    children=[
                        # html.P("POINT:"),
                        # dcc.RangeSlider(
                        #     id='point-slider',
                        #     min=0, max=125000, step=1000,
                        #     marks={0: '0', 125000: '125k'},
                        #     value=[0, 125000]
                        # ),
                        # html.P("SEGMENT:"),
                        # dcc.RangeSlider(
                        #     id='segment-slider',
                        #     min=0, max=150000, step=1000,
                        #     marks={0: '0', 150000: '150k'},
                        #     value=[0, 150000]
                        # ),
                        # html.P("GAIA:"),
                        # dcc.RangeSlider(
                        #     id='gaia-slider',
                        #     min=0, max=9000, step=100,
                        #     marks={0: '0', 9000: '9000'},
                        #     value=[0, 9000]
                        # )
                    ]
                ),
            ],
            style={
                "color": "white",
                "height": 800,
                "border": "2px #333 solid",
                "borderRadius": 5,
                "margin": 25,
                "padding": 10,
            },
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
