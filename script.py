from flask import Flask, render_template, request, url_for, redirect
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
    if request.method == "POST":
        userText = request.form["text"]

        # Save user message
        message = Message(text=userText)
        db.session.add(message)
        db.session.commit()

        # Process chatbot response
        response_text = chatBot.chat(message=message.text)
        response_message = Message(text=response_text)
        db.session.add(response_message)
        db.session.commit()

        # Redirect to refresh the page and use GET method
        return redirect(url_for("getStarted"))

        # GET method: Retrieve all messages
    messages = Message.query.all()
    return render_template('getStarted.html', messages=messages)

@app.route("/about")
def about():
    return render_template('AboutUs.html')

if __name__ == "__main__":
    app.run(debug=True)