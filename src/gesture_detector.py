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
    # Tap thresholds
    tap_min_met: float = 4.0
    tap_max_duration: int = 2
    
    # Rotation thresholds
    rotation_min_met: float = 1.5
    rotation_max_met: float = 4.0
    rotation_min_duration: int = 3
    rotation_merge_gap: int = 3
    
    # Hold thresholds
    hold_max_met: float = 1.2
    hold_min_duration: int = 8
    hold_stability_threshold: float = 0.3

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
        self.settings = settings or GestureSettings()

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
            met_indices = np.linspace(0, len(met_array)-1, len(motion_array)).astype(int)
            met_array = met_array[met_indices]
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
        
        i = 0
        while i < len(met_array) - 1:
            if met_array[i] >= self.settings.tap_min_met:
                end = i + 1
                while end < len(met_array) and met_array[end] >= self.settings.tap_min_met:
                    end += 1
                
                duration = end - i
                if duration <= self.settings.tap_max_duration:
                    taps.append(Gesture(
                        start_time=int(i),
                        end_time=int(end),
                        type=GestureType.TAP,
                        confidence=min(1.0, met_array[i] / self.settings.tap_min_met),
                        metadata={"met": float(met_array[i])}
                    ))
                i = end
            i += 1
        return taps

    def _detect_rotations(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        rotations = []
        i = 0
        
        while i < len(met_array):
            if (self.settings.rotation_min_met <= met_array[i] <= self.settings.rotation_max_met and 
                motion_array[i] == 3):
                start = i
                end = i + 1
                
                while end < len(met_array) and (
                    (self.settings.rotation_min_met <= met_array[end] <= self.settings.rotation_max_met and 
                     motion_array[end] == 3) or 
                    end - start <= self.settings.rotation_merge_gap
                ):
                    end += 1
                
                duration = end - start
                if duration >= self.settings.rotation_min_duration:
                    avg_met = np.mean(met_array[start:end])
                    rotations.append(Gesture(
                        start_time=int(start),
                        end_time=int(end),
                        type=GestureType.ROTATION,
                        confidence=min(1.0, duration / self.settings.rotation_min_duration),
                        metadata={
                            "duration": int(duration),
                            "avg_met": float(avg_met)
                        }
                    ))
                i = end
            i += 1
        
        return self._merge_rotations(rotations)

    def _merge_rotations(self, rotations: List[Gesture]) -> List[Gesture]:
        if not rotations:
            return []
            
        merged = []
        current = rotations[0]
        
        for next_rot in rotations[1:]:
            if next_rot.start_time - current.end_time <= self.settings.rotation_merge_gap:
                duration = next_rot.end_time - current.start_time
                avg_met = (current.metadata["avg_met"] * current.metadata["duration"] + 
                         next_rot.metadata["avg_met"] * next_rot.metadata["duration"]) / duration
                
                current = Gesture(
                    start_time=current.start_time,
                    end_time=next_rot.end_time,
                    type=GestureType.ROTATION,
                    confidence=max(current.confidence, next_rot.confidence),
                    metadata={
                        "duration": duration,
                        "avg_met": float(avg_met)
                    }
                )
            else:
                merged.append(current)
                current = next_rot
        
        merged.append(current)
        return merged

    def _detect_holds(self, motion_array: np.ndarray, met_array: np.ndarray) -> List[Gesture]:
        holds = []
        i = 0
        
        while i < len(met_array) - self.settings.hold_min_duration:
            window = motion_array[i:i + self.settings.hold_min_duration]
            window_met = met_array[i:i + self.settings.hold_min_duration]
            
            if (np.std(window) <= self.settings.hold_stability_threshold and 
                np.mean(window_met) <= self.settings.hold_max_met):
                
                end = i + self.settings.hold_min_duration
                while end < len(motion_array) and (
                    np.std(motion_array[i:end+1]) <= self.settings.hold_stability_threshold and
                    np.mean(met_array[i:end+1]) <= self.settings.hold_max_met
                ):
                    end += 1
                
                duration = end - i
                avg_met = np.mean(met_array[i:end])
                confidence = 1.0 - (avg_met / self.settings.hold_max_met)
                
                holds.append(Gesture(
                    start_time=int(i),
                    end_time=int(end),
                    type=GestureType.HOLD,
                    confidence=float(confidence),
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
            elif current.confidence > prev.confidence * 1.2:
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