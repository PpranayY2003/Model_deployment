from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your models
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("spam_classifier.html", email_text="", predictions="")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email = request.form.get('email-content')
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        prediction_text = "Spam" if prediction == 1 else "Not Spam"

    return render_template("spam_classifier.html", predictions=prediction_text, email_text=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
