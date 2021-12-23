import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import home, eval, eda, pred


url_bar_and_content_div = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div(
    [url_bar_and_content_div, home.layout, eval.layout, eda.layout, pred.layout]
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/layouts/home":
        return home.layout
    elif pathname == "/layouts/eval":
        return eval.layout
    elif pathname == "/layouts/eda":
        return eda.layout
    elif pathname == "/layouts/pred":
        return pred.layout
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=True)
