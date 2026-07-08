from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load(r"C:\Users\mayur\OneDrive\Desktop\data science training\project\cutomer churn predition system\model\log_model.lb")

# Load Training Columns
columns = joblib.load(r"C:\Users\mayur\OneDrive\Desktop\data science training\project\cutomer churn predition system\model\columns.lb")


@app.route('/')
def info():
    return render_template("info.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/basic')
def basic():
    return render_template("basic.html")


@app.route('/services')
def services():
    return render_template("service.html")


@app.route('/predict', methods=['POST'])
def predict():

    # ================= INPUT =================

    gender = int(request.form['gender'])
    SeniorCitizen = int(request.form['SeniorCitizen'])
    Partner = int(request.form['Partner'])
    Dependents = int(request.form['Dependents'])
    tenure = int(request.form['tenure'])
    PhoneService = int(request.form['PhoneService'])
    MultipleLines = int(request.form['MultipleLines'])
    InternetService = int(request.form['InternetService'])

    OnlineSecurity = request.form['OnlineSecurity']
    OnlineBackup = request.form['OnlineBackup']
    DeviceProtection = request.form['DeviceProtection']
    TechSupport = request.form['TechSupport']
    StreamingTV = request.form['StreamingTV']
    StreamingMovies = request.form['StreamingMovies']

    Contract = int(request.form['Contract'])
    PaperlessBilling = int(request.form['PaperlessBilling'])
    PaymentMethod = int(request.form['PaymentMethod'])
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])

    # ================= ENCODING =================

    os_no = int(OnlineSecurity == "No")
    os_yes = int(OnlineSecurity == "Yes")
    os_no_internet = int(OnlineSecurity == "No Internet Service")

    ob_no = int(OnlineBackup == "No")
    ob_yes = int(OnlineBackup == "Yes")
    ob_no_internet = int(OnlineBackup == "No Internet Service")

    dp_no = int(DeviceProtection == "No")
    dp_yes = int(DeviceProtection == "Yes")
    dp_no_internet = int(DeviceProtection == "No Internet Service")

    ts_no = int(TechSupport == "No")
    ts_yes = int(TechSupport == "Yes")
    ts_no_internet = int(TechSupport == "No Internet Service")

    stv_no = int(StreamingTV == "No")
    stv_yes = int(StreamingTV == "Yes")
    stv_no_internet = int(StreamingTV == "No Internet Service")

    sm_no = int(StreamingMovies == "No")
    sm_yes = int(StreamingMovies == "Yes")
    sm_no_internet = int(StreamingMovies == "No Internet Service")

    # ================= INPUT DATAFRAME =================

    input_data = pd.DataFrame([[

        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,

        os_no,
        os_no_internet,
        os_yes,

        ob_no,
        ob_yes,
        ob_no_internet,

        dp_no,
        dp_yes,
        dp_no_internet,

        ts_no,
        ts_yes,
        ts_no_internet,

        stv_no,
        stv_yes,
        stv_no_internet,

        sm_no,
        sm_yes,
        sm_no_internet,

        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges

    ]])

    # ================= COLUMN ALIGN =================

    input_data = input_data.reindex(columns=columns, fill_value=0)

    # ================= PREDICTION =================

    prediction = model.predict(input_data)[0]

    churn_prob = model.predict_proba(input_data)[0][1] * 100
    stay_prob = 100 - churn_prob

    if prediction == 1:
        result = "Customer Will Leave"
        probability = churn_prob
    else:
        result = "Customer Will Stay"
        probability = stay_prob

    # ================= RISK LEVEL =================

    if prediction == 0:
        risk = "Low Risk"
    else:
        if churn_prob >= 80:
            risk = "High Risk"
        elif churn_prob >= 50:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

    # ================= RETURN =================

    return render_template(
        "churn.html",
        prediction=result,
        probability=round(probability, 2),
        risk=risk
    )


if __name__ == "__main__":
    app.run(debug=True)