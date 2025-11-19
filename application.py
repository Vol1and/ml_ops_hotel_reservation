import numpy as np

from config.paths_config import LGMB_MODEL_PATH
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

loaded_model = joblib.load(LGMB_MODEL_PATH)


@app.route('/', methods=['POST'])
def index():
    request_json = request.get_json()
    no_of_adults = int(request_json['no_of_adults'])
    no_of_children = int(request_json['no_of_children'])
    no_of_weekend_nights = int(request_json['no_of_weekend_nights'])
    no_of_week_nights = int(request_json['no_of_week_nights'])
    type_of_meal_plan = int(request_json['type_of_meal_plan'])
    required_car_parking_space = int(request_json['required_car_parking_space'])
    room_type_reserved = int(request_json['room_type_reserved'])
    lead_time = int(request_json['lead_time'])
    arrival_year = int(request_json['arrival_year'])
    arrival_month = int(request_json['arrival_month'])

    features = np.array([
        no_of_adults,
        no_of_children,
        no_of_weekend_nights,
        no_of_week_nights,
        type_of_meal_plan,
        required_car_parking_space,
        room_type_reserved,
        lead_time,
        arrival_year,
        arrival_month])

    prediction = loaded_model.predict([features])

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)







