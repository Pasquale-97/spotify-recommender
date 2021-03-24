from application import app


@app.route("/")
def index():
    return 'flask app is running'

