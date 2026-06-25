
from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)


model = pickle.load(open("models/model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from HTML
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array
        final_input = np.array([features])

        # Predict HDI
        prediction = model.predict(final_input)

        # Round result
        output = round(prediction[0], 2)

        return render_template("result.html", prediction_text=f"Predicted HDI Score: {output}")

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)