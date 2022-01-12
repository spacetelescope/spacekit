from dash import dcc
from dash import html

layout = html.Div(
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
