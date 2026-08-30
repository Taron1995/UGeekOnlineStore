"""Flask dashboard for reviewing logged smoking-detection events."""
import os
import sys

from flask import Flask, render_template, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SNAPSHOTS_DIR
from src import db

app = Flask(__name__)
db.init_db()


@app.template_filter("timestamp")
def format_timestamp(value):
    import datetime
    if not value:
        return "—"
    return datetime.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


@app.route("/")
def index():
    summary = db.detections_per_employee()
    recent = db.list_detections(limit=25)
    return render_template("index.html", summary=summary, recent=recent)


@app.route("/employees")
def employees():
    rows = db.list_employees()
    return render_template("employees.html", employees=rows)


@app.route("/history")
def history():
    rows = db.list_detections(limit=1000)
    return render_template("history.html", detections=rows)


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
