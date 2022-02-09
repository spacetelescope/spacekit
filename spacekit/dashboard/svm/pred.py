# TODO: pick a random visit - show the three image frames, make prediction
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
        )
    ],
    style={
        "width": "100%",
        "height": "100%",
        "background-color": "#242a44",
        "color": "white",
    },
)
