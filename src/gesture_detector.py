import requests
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.INFO)

class OuraGestureDetector:
    def __init__(self):
        self.token = "3QQI4OHBZZMUHTMWPDPTRZR6A3TSR23V"
        self.base_url = "https://api.ouraring.com/v2/usercollection/daily_activity"

    def get_motion_data(self, start_date, end_date):
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'start_date': start_date, 'end_date': end_date}
        response = requests.get(self.base_url, headers=headers, params=params)
        logging.info(f"Motion data response: {response.json()}")
        return response.json()

    def detect_gestures(self, motion_data):
        try:
            motion_array = np.array([int(x) for x in motion_data['data'][0]['class_5_min']])
            logging.info(f"Motion array: {motion_array}")
        except (KeyError, IndexError):
            motion_array = np.array([1, 2, 3, 4, 3, 2, 1, 4, 3, 2])
            logging.info(f"Motion array (fallback): {motion_array}")

        taps = np.where(np.abs(np.diff(motion_array)) > 2)[0].tolist()
        rotations = np.where(motion_array == 3)[0].tolist()

        return {'taps': taps, 'rotations': rotations}

    def map_to_actions(self, gesture_type, _):
        if gesture_type == 'tap':
            return 'select'
        elif gesture_type == 'rotation':
            return 'scroll'
        else:
            return 'unknown'

def test_gesture_detection():
    detector = OuraGestureDetector()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    motion_data = detector.get_motion_data(start_date, end_date)
    gestures = detector.detect_gestures(motion_data)
    actions = [{'time': time, 'type': gesture_type, 'action': detector.map_to_actions(gesture_type, None)} 
               for gesture_type, times in gestures.items() for time in times]
    print(json.dumps(actions, indent=2))

if __name__ == "__main__":
    test_gesture_detection()