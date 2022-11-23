import process
from flask import Flask, render_template, request

#Start Chatbot
app = Flask(__name__)

#import process
modelObject = process.ModelClass()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    print(type(request.args.get('msg')))
    result = modelObject.chatbot_response(text=user_input)
    return result

if __name__ == "__main__":
    app.run(debug=True)