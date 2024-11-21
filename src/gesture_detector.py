import numpy as np
from datetime import datetime, timedelta
import requests
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)

class GestureType(Enum):
    TAP = "tap"
    ROTATION = "rotation"
    DOUBLE_TAP = "double_tap"
    HOLD = "hold"

@dataclass
class Gesture:
    start_time: int
    end_time: Optional[int]
    type: GestureType
    confidence: float
    metadata: Dict

class OuraGestureDetector:
    def __init__(self, token: str):
        self.token = "3QQI4OHBZZMUHTMWPDPTRZR6A3TSR23V"
        self.base_url = "https://api.ouraring.com/v2/usercollection/daily_activity"
        
        # Gesture detection parameters
        self.tap_threshold = 2
        self.rotation_threshold = 3
        self.double_tap_max_gap = 3  # max time units between taps
        self.hold_min_duration = 4    # min time units for hold
        
    def get_motion_data(self, start_date: str, end_date: str) -> Dict:
        headers = {'Authorization': f'Bearer {self.token}'}
        params = {'start_date': start_date, 'end_date': end_date}
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch motion data: {e}")
            return {}

    def detect_gestures(self, motion_data: Dict) -> List[Gesture]:
        try:
            motion_array = np.array([int(x) for x in motion_data['data'][0]['class_5_min']])
            met_array = np.array(motion_data['data'][0]['met']['items'])
        except (KeyError, IndexError) as e:
            logging.error(f"Failed to process motion data: {e}")
            return []

        gestures = []
        gestures.extend(self._detect_taps(motion_array, met_array))
        gestures.extend(self._detect_rotations(motion_array, met_array))
        gestures.extend(self._detect_double_taps(motion_array, met_array))
        gestures.extend(self._detect_holds(motion_array, met_array))
        
        return self._remove_overlapping_gestures(gestures)

    def _detect_taps(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        taps = []
        diffs = np.abs(np.diff(motion_array))
        tap_indices = np.where(diffs > self.tap_threshold)[0]
        
        for idx in tap_indices:
            confidence = min(1.0, diffs[idx] / (self.tap_threshold * 2))
            met_value = met_array[idx] if idx < len(met_array) else 0
            
            taps.append(Gesture(
                start_time=int(idx),
                end_time=int(idx + 1),
                type=GestureType.TAP,
                confidence=float(confidence),
                metadata={"met": float(met_value)}
            ))
        
        return taps

    def _detect_rotations(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        rotations = []
        rotation_mask = motion_array == self.rotation_threshold
        rotation_starts = np.where(np.diff(rotation_mask.astype(int)) == 1)[0]
        rotation_ends = np.where(np.diff(rotation_mask.astype(int)) == -1)[0]
        
        for start, end in zip(rotation_starts, rotation_ends):
            duration = end - start
            confidence = min(1.0, duration / 3)  # Longer rotations = higher confidence
            avg_met = np.mean(met_array[start:end+1]) if start < len(met_array) else 0
            
            rotations.append(Gesture(
                start_time=int(start),
                end_time=int(end),
                type=GestureType.ROTATION,
                confidence=float(confidence),
                metadata={
                    "duration": int(duration),
                    "avg_met": float(avg_met)
                }
            ))
        
        return rotations

    def _detect_double_taps(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        taps = self._detect_taps(motion_array, met_array)
        double_taps = []
        
        for i in range(len(taps) - 1):
            if taps[i+1].start_time - taps[i].end_time <= self.double_tap_max_gap:
                confidence = min(taps[i].confidence, taps[i+1].confidence)
                double_taps.append(Gesture(
                    start_time=taps[i].start_time,
                    end_time=taps[i+1].end_time,
                    type=GestureType.DOUBLE_TAP,
                    confidence=confidence,
                    metadata={"gap": taps[i+1].start_time - taps[i].end_time}
                ))
        
        return double_taps

    def _detect_holds(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        holds = []
        stable_regions = np.where(np.abs(np.diff(motion_array)) <= 0.5)[0]
        
        if len(stable_regions) == 0:
            return holds
            
        # Group consecutive stable regions
        hold_starts = [stable_regions[0]]
        hold_ends = []
        
        for i in range(1, len(stable_regions)):
            if stable_regions[i] - stable_regions[i-1] > 1:
                hold_ends.append(stable_regions[i-1])
                hold_starts.append(stable_regions[i])
        hold_ends.append(stable_regions[-1])
        
        for start, end in zip(hold_starts, hold_ends):
            duration = end - start
            if duration >= self.hold_min_duration:
                confidence = min(1.0, duration / (self.hold_min_duration * 2))
                avg_met = np.mean(met_array[start:end+1]) if start < len(met_array) else 0
                
                holds.append(Gesture(
                    start_time=int(start),
                    end_time=int(end),
                    type=GestureType.HOLD,
                    confidence=float(confidence),
                    metadata={
                        "duration": int(duration),
                        "avg_met": float(avg_met)
                    }
                ))
        
        return holds

    def _remove_overlapping_gestures(self, gestures: List[Gesture]) -> List[Gesture]:
        if not gestures:
            return []
            
        gestures.sort(key=lambda x: (x.start_time, -x.confidence))
        result = [gestures[0]]
        
        for current in gestures[1:]:
            prev = result[-1]
            if current.start_time >= prev.end_time:
                result.append(current)
            elif current.confidence > prev.confidence:
                result[-1] = current
                
        return result

def test_gesture_detection():
    detector = OuraGestureDetector(token="YOUR_TOKEN")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    motion_data = detector.get_motion_data(start_date, end_date)
    gestures = detector.detect_gestures(motion_data)
    
    print(json.dumps([{
        "start_time": g.start_time,
        "end_time": g.end_time,
        "type": g.type.value,
        "confidence": g.confidence,
        "metadata": g.metadata
    } for g in gestures], indent=2))

if __name__ == "__main__":
    test_gesture_detection()