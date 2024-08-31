from flask import Flask, render_template, request
import pickle
import pandas as pd
from Weather import data1
import datetime
import pytz

app = Flask(__name__, static_url_path='/static')

# Load the pre-trained machine learning models
with open("XGB_model.pkl", 'rb') as file:
    XGB_model = pickle.load(file)
with open("XGB_model_distance.pkl", 'rb') as file1:
    XGB_model_distance = pickle.load(file1)


def date_time():
    current_time = datetime.datetime.now(pytz.timezone('America/New_York'))
    return current_time


# label encoded values for html page drop-down input mapping
class CustomData:
    def __init__(self, Current_Location: str, Destination: str, Car_type: str, Rain_level: float):
        location_mapping = {
            'Back Bay': 0,
            'Beacon Hill': 1,
            'Boston University': 2,
            'Fenway': 3,
            'Financial District': 4,
            'Haymarket Square': 5,
            'North End': 6,
            'North Station': 7,
            'Northeastern University': 8,
            'South Station': 9,
            'Theatre District': 10,
            'West End': 11
        }

        car_type_mapping = {
            'Black': 0,
            'Black SUV': 1,
            'UberPool': 2,
            'UberX': 3,
            'UberXL': 4,
            'WAV': 5
        }

        self.Current_Location = location_mapping.get(Current_Location, 11)
        self.Destination = location_mapping.get(Destination, 11)
        self.Car_type = car_type_mapping.get(Car_type, 5)
        self.Rain_level = Rain_level

    def get_data_as_data_frame(self):
        current_time = date_time()
        custom_data_input_dict = {
            "destination": [self.Destination],
            "source": [self.Current_Location],
            "name": [self.Car_type],
            "temp": [data1["main"]["temp"] - 273],  # Convert from Kelvin to Celsius
            "location": [self.Current_Location],
            "clouds": [data1.get("clouds", {}).get("all", 0)],
            "pressure": [data1.get("main", {}).get("pressure", 0)],
            "rain": [self.Rain_level],
            "humidity": [data1.get("main", {}).get("humidity", 0) / 100],
            "wind": [data1.get("wind", {}).get("speed", 0)],
            "Wweek": [current_time.day % 7],
            "Whour": [current_time.hour],
            "datetime": [current_time.timestamp()],
            "Wdate": [(current_time - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)).days],
        }
        return pd.DataFrame(custom_data_input_dict)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    data = CustomData(
        Current_Location=request.form.get('Current_Location'),
        Destination=request.form.get('Destination'),
        Car_type=request.form.get('Car_type'),
        Rain_level=float(request.form.get('Rain_level'))
    )
    pred_df1 = data.get_data_as_data_frame()
    pred_df2 = data.get_data_as_data_frame()

    # Ensure pred_df1 has the 'rain' column for distance prediction
    if 'rain' not in pred_df1.columns:
        raise ValueError("The feature 'rain' is missing from the input data.")

    # Drop unnecessary columns for distance prediction
    pred_df1 = pred_df1[['destination', 'source','rain']]

    # Predict distance using the first model
    distance = XGB_model_distance.predict(pred_df1)[0]
    print("distance = ",distance)
    # Add the distance to the second dataframe
    pred_df2.insert(0, 'distance', distance)

    # Ensure pred_df2 has the correct order of columns expected by the second model
    expected_columns = ['distance', 'destination', 'source', 'name', 'temp', 'location', 'clouds',
                        'pressure', 'rain', 'humidity', 'wind']
    pred_df2 = pred_df2[expected_columns]

    # Predict price using the second model
    predicted_price = round(XGB_model.predict(pred_df2)[0], 2)

    return render_template('index.html', results=predicted_price)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
