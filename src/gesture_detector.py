import requests
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import json

class OuraGestureDetector:
    def __init__(self):
        self.client_id = "RWCGPZCWJ4I4AXZP"
        self.client_secret = "4YUWDBU3OGJ2HAAOVYVAJXNVVOYPGVIR"
        self.token_url = "https://api.ouraring.com/oauth/token"
        self.base_url = "https://api.ouraring.com/v2/usercollection/daily_activity"
        
        # Get OAuth token
        self.token = self._get_token()
        
    def _get_token(self):
        client = BackendApplicationClient(client_id=self.client_id)
        oauth = OAuth2Session(client=client)
        return oauth.fetch_token(
            token_url=self.token_url,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
    def get_motion_data(self, start_date, end_date):
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'start_date': start_date, 'end_date': end_date}
        response = requests.get(self.base_url, headers=headers, params=params)
        return response.json()
    
    def detect_gestures(self, motion_data):
        # Convert motion data to numpy array
        # For testing, create dummy data if no real data available
        try:
            motion_array = np.array([int(x) for x in motion_data['data'][0]['class_5_min']])
        except (KeyError, IndexError):
            motion_array = np.array([1,2,3,4,3,2,1,4,3,2]) # Dummy data
        
        # Detect peaks for taps
        peaks, _ = find_peaks(motion_array, height=3, distance=2)
        
        # Detect rotations (sustained medium activity)
        rotations = np.where(motion_array == 3)[0]
        
        # Detect quick gestures (rapid changes)
        gestures = np.where(np.diff(motion_array) > 1)[0]
        
        return {
            'taps': peaks.tolist(),
            'rotations': rotations.tolist(),
            'gestures': gestures.tolist()
        }
    
    def map_to_actions(self, gesture_type, count):
        actions = {
            'tap': {
                1: 'select',
                2: 'back'
            },
            'rotation': {
                1: 'scroll',
                2: 'zoom'
            },
            'gesture': {
                1: 'next',
                2: 'previous'
            }
        }
        return actions.get(gesture_type, {}).get(count, 'unknown')

def test_gesture_detection():
    # Initialize with sandbox token
    detector = OuraGestureDetector("SANDBOX_TOKEN")
    
    # Get today and yesterday's dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Get test data
    motion_data = detector.get_motion_data(start_date, end_date)
    
    # Run gesture detection
    gestures = detector.detect_gestures(motion_data)
    
    # Map gestures to actions
    actions = []
    for g_type, times in gestures.items():
        for t in times:
            action = detector.map_to_actions(g_type, len(times))
            actions.append({
                'time': t,
                'type': g_type,
                'action': action
            })
    
    print("Detected gestures and mapped actions:")
    print(json.dumps(actions, indent=2))

if __name__ == "__main__":
    test_gesture_detection()