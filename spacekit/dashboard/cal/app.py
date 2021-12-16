import dash

app = dash.Dash(
    __name__,
    title="spacekit dashboard",
    assets_folder="../assets",
    suppress_callback_exceptions=True,
)

server = app.server
