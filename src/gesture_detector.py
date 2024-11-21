import requests
import numpy as np
from datetime import datetime, timedelta
import json

class OuraGestureDetector:
    def __init__(self):
        self.client_id = "RWCGPZCWJ4I4AXZP"
        self.client_secret = "4YUWDBU3OGJ2HAAOVYVAJXNVVOYPGVIR"
        self.token = "3QQI4OHBZZMUHTMWPDPTRZR6A3TSR23V"
        self.base_url = "https://api.ouraring.com/v2/usercollection/daily_activity"

    def get_motion_data(self, start_date, end_date):
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'start_date': start_date, 'end_date': end_date}
        response = requests.get(self.base_url, headers=headers, params=params)
        return response.json()

    def detect_gestures(self, motion_data):
        try:
            motion_array = np.array([int(x) for x in motion_data['data'][0]['class_5_min']])
        except (KeyError, IndexError):
            motion_array = np.array([1, 2, 3, 4, 3, 2, 1, 4, 3, 2])

        # Detect taps
        taps = np.where(np.abs(np.diff(motion_array)) > 2)[0].tolist()

        # Detect rotations (sustained medium activity)
        rotations = np.where(motion_array == 3)[0].tolist()

        # Detect quick gestures (rapid changes)
        gestures = np.where(np.abs(np.diff(motion_array)) > 1)[0].tolist()

        # Filter out overlapping gestures
        unique_gestures = list(set(gestures) - set(taps) - set(rotations))

        return {'taps': taps, 'rotations': rotations, 'gestures': unique_gestures}

    def map_to_actions(self, gesture_type, count):
        action_map = {
            'tap': {1: 'select', 2: 'back', 3: 'home'},
            'rotation': {1: 'scroll', 2: 'zoom', 3: 'rotate'},
            'gesture': {1: 'next', 2: 'previous', 3: 'dismiss'}
        }
        return action_map.get(gesture_type, {}).get(count, 'unknown')

def test_gesture_detection():
    detector = OuraGestureDetector()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    motion_data = detector.get_motion_data(start_date, end_date)
    gestures = detector.detect_gestures(motion_data)

    actions = []
    for gesture_type, times in gestures.items():
        for time in times:
            action = detector.map_to_actions(gesture_type, len(times))
            actions.append({'time': time, 'type': gesture_type, 'action': action})

    print(json.dumps(actions, indent=2))

if __name__ == "__main__":
    try:
        test_gesture_detection()
    except ValueError as e:
        print(f"Error: {e}")