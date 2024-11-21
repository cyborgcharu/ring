# src/gesture_detector.py
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
    HOLD = "hold"

@dataclass
class GestureSettings:
    tap_threshold: float = 2.0
    met_activity_threshold: float = 1.5
    hold_stability_threshold: float = 0.5
    hold_min_duration: int = 8
    rotation_min_duration: int = 2
    rotation_max_gap: int = 1

@dataclass
class Gesture:
    start_time: int
    end_time: Optional[int]
    type: GestureType
    confidence: float
    metadata: Dict

class OuraGestureDetector:
    def __init__(self, settings: GestureSettings = None):
        self.token = "3QQI4OHBZZMUHTMWPDPTRZR6A3TSR23V"
        self.base_url = "https://api.ouraring.com/v2/usercollection/daily_activity"
        
        settings = settings or GestureSettings()
        self.tap_threshold = settings.tap_threshold
        self.rotation_min_duration = settings.rotation_min_duration
        self.rotation_max_gap = settings.rotation_max_gap
        self.hold_min_duration = settings.hold_min_duration
        self.hold_stability_threshold = settings.hold_stability_threshold
        self.met_activity_threshold = settings.met_activity_threshold

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
            
            # Resample met_array to match motion_array length
            met_indices = np.linspace(0, len(met_array)-1, len(motion_array)).astype(int)
            met_array = met_array[met_indices]
            
            logging.info(f"Motion array length: {len(motion_array)}, MET array length: {len(met_array)}")
        except (KeyError, IndexError) as e:
            logging.error(f"Failed to process motion data: {e}")
            return []

        gestures = []
        gestures.extend(self._detect_taps(motion_array, met_array))
        gestures.extend(self._detect_rotations(motion_array, met_array))
        gestures.extend(self._detect_holds(motion_array, met_array))
        
        return self._remove_overlapping_gestures(gestures)

    def _detect_taps(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        taps = []
        diffs = np.abs(np.diff(motion_array))
        met_mask = met_array[:-1] > self.met_activity_threshold
        tap_indices = np.where((diffs > self.tap_threshold) & met_mask)[0]
        
        for idx in tap_indices:
            confidence = min(1.0, (diffs[idx] / self.tap_threshold) * (met_array[idx] / self.met_activity_threshold))
            taps.append(Gesture(
                start_time=int(idx),
                end_time=int(idx + 1),
                type=GestureType.TAP,
                confidence=float(confidence),
                metadata={"met": float(met_array[idx])}
            ))
        return taps

    def _detect_rotations(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        rotations = []
        rotation_mask = motion_array == 3
        
        i = 0
        while i < len(rotation_mask):
            if rotation_mask[i]:
                start = i
                while i < len(rotation_mask) and (rotation_mask[i] or i - start <= self.rotation_max_gap):
                    i += 1
                end = i
                
                duration = end - start
                if duration >= self.rotation_min_duration:
                    avg_met = np.mean(met_array[start:end])
                    if avg_met > self.met_activity_threshold:
                        confidence = min(1.0, duration / self.rotation_min_duration * (avg_met / self.met_activity_threshold))
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
            i += 1
        return rotations

    def _detect_holds(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        holds = []
        i = 0
        while i < len(motion_array) - self.hold_min_duration:
            window = motion_array[i:i + self.hold_min_duration]
            if np.std(window) <= self.hold_stability_threshold:
                end = i + self.hold_min_duration
                while end < len(motion_array) and np.std(motion_array[i:end+1]) <= self.hold_stability_threshold:
                    end += 1
                
                duration = end - i
                avg_met = np.mean(met_array[i:end])
                stability = 1 - (np.std(motion_array[i:end]) / self.hold_stability_threshold)
                
                holds.append(Gesture(
                    start_time=int(i),
                    end_time=int(end),
                    type=GestureType.HOLD,
                    confidence=float(stability),
                    metadata={
                        "duration": int(duration),
                        "avg_met": float(avg_met)
                    }
                ))
                i = end
            i += 1
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
    detector = OuraGestureDetector()
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