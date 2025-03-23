from flask import Flask, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy
import AI

chatBot = AI.MentalHealthChatbot(
    data_path='test2.json',
    model_dir='model_output',
    load_existing=True,
    epochs=0
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blog.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return '<Message %r>' % self.id

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/getStarted", methods=["POST", "GET"])
def getStarted():
    messages = Message.query.all()

    if request.method == "POST":
        userText = request.form["text"]

        message = Message(text=userText)

        db.session.add(message)
        db.session.commit()

        message = Message(text=chatBot.chat(message=message.text))

        db.session.add(message)
        db.session.commit()
    return render_template('getStarted.html')

@app.route("/about")
def about():
    return render_template('AboutUs.html')

if __name__ == "__main__":
    app.run(debug=True)