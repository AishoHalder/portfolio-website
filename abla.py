from flask import Flask,render_template
from facefunstions import imp
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/a2")
def a2():
    x = imp.face_recog()
    if x >= 13 :
        return render_template("a2.html")
    return render_template("main.html")


if __name__ == '__main__':
    app.run(debug=True)