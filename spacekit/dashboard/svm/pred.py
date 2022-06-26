# TODO: pick a random visit - show the three image frames+data, augmented images+data, make prediction
from dash import dcc
from dash import html

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
                        dcc.Dropdown(
                            id="selected-image",
                            options=[
                                {"label": i, "value": i}
                                for i in [
                                    "img1",
                                    "img2",
                                ]
                            ],
                            value="img1",
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
                            id="image-graph",
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
    ],
    style={
        "width": "100%",
        "height": "100%",
        "background-color": "#242a44",
        "color": "white",
    },
)
