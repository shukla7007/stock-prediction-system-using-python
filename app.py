import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained models and data
data = pd.read_csv("TATAMOTORS_BSE_01Oct20-01Oct21.csv")  # Replace with your data file path
data.replace('null', pd.NA, inplace=True)
data = data.dropna(axis=0)

y = data['Close']
X = data.drop(['Close', 'Date', 'Volume', 'Adj Close'], axis=1)

reg = LinearRegression().fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        mhigh = float(request.form['market_high'])
        mlow = float(request.form['market_low'])
        mopen = float(request.form['market_open'])

        x_new = [[mhigh, mlow, mopen]]
        prediction = reg.predict(x_new)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
