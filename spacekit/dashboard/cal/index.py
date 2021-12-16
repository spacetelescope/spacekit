import argparse
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import home, eval, eda, pred
from spacekit.skopes.hst.cal.datasets import get_sample_data, scrape_s3

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


# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", choices=["file", "git", "s3"], default="file")
    parser.add_argument(
        "-u", "--uri", default="data", help="s3 bucketname, filepath, or web uri"
    )
    parser.add_argument("-l", "--latest", default=0, help="scrape last 3 timestamps")
    parser.add_argument("-r0", "--results1", default="2021-11-04-1636048291")
    parser.add_argument("-r1", "--results2", default="2021-10-28-1635457222")
    parser.add_argument("-r2", "--results3", default="2021-08-22-1629663047")
    args = parser.parse_args()
    if args.src == "git":
        fpaths = get_sample_data()
        print(fpaths)
    elif args.src == "s3":
        scrape_s3(args.uri, get_latest=args.latest, results=[args.r0, args.r1, args.r2])
    app.run_server(host="0.0.0.0", port=8050, debug=True)
