from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model = joblib.load('student_performance_model.pkl')

columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", 
    "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"
]

categorical_columns = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", 
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"
]

encoders = {col: LabelEncoder() for col in categorical_columns}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    prediction = None  
    if request.method == "POST":
        data = [
            [
                request.form["school"],
                request.form["sex"],
                int(request.form["age"]),
                request.form["address"],
                request.form["famsize"],
                request.form["Pstatus"],
                int(request.form["Medu"]),
                int(request.form["Fedu"]),
                request.form["Mjob"],
                request.form["Fjob"],
                request.form["reason"],
                request.form["guardian"],
                int(request.form["traveltime"]),
                int(request.form["studytime"]),
                int(request.form["failures"]),
                request.form["schoolsup"],
                request.form["famsup"],
                request.form["paid"],
                request.form["activities"],
                request.form["nursery"],
                request.form["higher"],
                request.form["internet"],
                request.form["romantic"],
                int(request.form["famrel"]),
                int(request.form["freetime"]),
                int(request.form["goout"]),
                int(request.form["Dalc"]),
                int(request.form["Walc"]),
                int(request.form["health"]),
                int(request.form["absences"]),
                int(request.form["G1"]),
                int(request.form["G2"]),
            ]
        ]

        df = pd.DataFrame(data, columns=columns[:-1])

        for col in categorical_columns:
            if col in df.columns:
                df[col] = encoders[col].fit_transform(df[col])

        prediction = model.predict(df)[0]  

        prediction = int(prediction) 

        print(prediction)

        return jsonify(result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
