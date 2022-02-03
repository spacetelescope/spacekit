# TODO: pick a random visit - show the three image frames, make prediction
from dash import dcc
from dash import html
# from dash.dependencies import Input, Output, State

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
                )],
    style={
        "width": "100%",
        "height": "100%",
        "background-color": "#242a44",
        "color": "white",
    },
)