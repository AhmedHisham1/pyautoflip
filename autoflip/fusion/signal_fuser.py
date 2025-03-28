from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


class StandardSignalType(Enum):
    """Standard signal types similar to MediaPipe AutoFlip."""
    FACE_CORE_LANDMARKS = 0
    FACE_ALL_LANDMARKS = 1
    FACE_FULL = 2
    HUMAN = 3
    PET = 4
    CAR = 5
    OBJECT = 6
    TEXT = 7
    USER_HINT = 8


@dataclass
class SignalType:
    """Signal type with standard or custom identifier."""
    standard: Optional[StandardSignalType] = None
    custom: Optional[str] = None
    
    def __str__(self):
        if self.standard is not None:
            return f"standard_{self.standard.name}"
        else:
            return f"custom_{self.custom}"


@dataclass
class SignalSettings:
    """Settings for a specific signal type."""
    type: SignalType
    min_score: float = 0.0
    max_score: float = 1.0
    is_required: bool = False


@dataclass
class SalientRegion:
    """Detected salient region with metadata."""
    x: float
    y: float
    width: float
    height: float
    score: float
    signal_type: SignalType
    tracking_id: Optional[int] = None
    is_required: bool = False
    
    def get(self, key, default=None):
        """Get attribute by name for dictionary-like access."""
        if key == 'x':
            return self.x
        elif key == 'y':
            return self.y
        elif key == 'width':
            return self.width
        elif key == 'height':
            return self.height
        elif key == 'confidence':
            return self.score
        elif key == 'score':
            return self.score
        elif key == 'tracking_id':
            return self.tracking_id
        elif key == 'is_required':
            return self.is_required
        elif key == 'class':
            # Figure out class from signal_type
            if self.signal_type.standard == StandardSignalType.FACE_FULL:
                return 'face'
            elif self.signal_type.standard == StandardSignalType.HUMAN:
                return 'person'
            elif self.signal_type.standard == StandardSignalType.PET:
                return 'animal'
            elif self.signal_type.standard == StandardSignalType.CAR:
                return 'car'
            else:
                return 'object'
        return default
    
    def copy(self):
        """Create a copy of this region."""
        return SalientRegion(
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            score=self.score,
            signal_type=self.signal_type,
            tracking_id=self.tracking_id,
            is_required=self.is_required
        )


@dataclass
class InputSignal:
    """Input signal with source information."""
    signal: SalientRegion
    source: int


@dataclass
class Frame:
    """Frame with input detections and timestamp."""
    input_detections: List[InputSignal]
    time_ms: int


class SignalFuser:
    """
    Fuses detection signals from multiple sources.
    
    This class combines detections from different sources (face, object, etc.)
    with configurable weighting and priority, closely following MediaPipe's
    SignalFusingCalculator.
    """
    
    def __init__(
        self,
        signal_settings: List[SignalSettings] = None,
        process_by_scene: bool = True
    ):
        """
        Initialize the signal fuser.
        
        Args:
            signal_settings: Configuration for each signal type
            process_by_scene: If True, accumulate frames for scene-level processing
        """
        self.process_by_scene = process_by_scene
        
        # Initialize settings by type dictionary
        self.settings_by_type = {}
        
        if signal_settings:
            for settings in signal_settings:
                self.settings_by_type[str(settings.type)] = settings
        
        # Initialize buffer for scene processing
        self.scene_frames = []
        
    def add_frame(
        self,
        detections_by_source: Dict[int, List[SalientRegion]],
        time_ms: int
    ):
        """
        Add a frame with detections to the processing queue.
        
        Args:
            detections_by_source: Dictionary mapping source IDs to detection lists
            time_ms: Timestamp in milliseconds
            
        Returns:
            Processed detections if processing immediately, None if buffering
        """
        frame = Frame(input_detections=[], time_ms=time_ms)
        
        # Add all detections to the frame
        for source_id, detections in detections_by_source.items():
            for detection in detections:
                input_signal = InputSignal(signal=detection, source=source_id)
                frame.input_detections.append(input_signal)
                
        self.scene_frames.append(frame)
        
        # Process immediately if not using scene-based processing
        if not self.process_by_scene:
            return self.process_scene()
            
        return None

    def on_shot_boundary(self) -> Dict[int, List[SalientRegion]]:
        """
        Process all buffered frames on shot boundary and return results.
        
        Returns:
            Dictionary mapping timestamps to fused detection lists
        """
        if not self.scene_frames:
            return {}
            
        result = self.process_scene()
        self.scene_frames.clear()
        return result
        
    def process_scene(self) -> Dict[int, List[SalientRegion]]:
        """
        Process all buffered frames and return fused detections.
        
        Returns:
            Dictionary mapping timestamps to fused detection lists
        """
        if not self.scene_frames:
            return {}
            
        detection_count = {}
        multiframe_score = {}
        
        # First pass: calculate average scores for tracked detections
        for frame in self.scene_frames:
            for detection in frame.input_detections:
                if detection.signal.tracking_id is not None:
                    # Create a unique key for this detection
                    key = f"{detection.source}:{detection.signal.tracking_id}"
                    
                    if key not in detection_count:
                        multiframe_score[key] = 0.0
                        detection_count[key] = 0
                        
                    multiframe_score[key] += detection.signal.score
                    detection_count[key] += 1
        
        # Calculate average scores
        for key in multiframe_score:
            multiframe_score[key] = multiframe_score[key] / detection_count[key]
            
        # Second pass: process each detection and apply settings
        results = {}
        for frame in self.scene_frames:
            processed_detections = []
            
            for input_detection in frame.input_detections:
                detection = input_detection.signal
                
                # Use multi-frame score if available for this detection
                score = detection.score
                if detection.tracking_id is not None:
                    key = f"{input_detection.source}:{detection.tracking_id}"
                    if key in multiframe_score:
                        score = multiframe_score[key]
                
                # Apply settings if available for this type
                type_key = str(detection.signal_type)
                if type_key in self.settings_by_type:
                    settings = self.settings_by_type[type_key]
                    
                    # Rescale score within specified range
                    min_value = settings.min_score
                    max_value = settings.max_score
                    final_score = score * (max_value - min_value) + min_value
                    
                    # Apply required flags
                    detection.is_required = settings.is_required
                    
                    # Update score
                    detection.score = final_score
                
                processed_detections.append(detection)
            
            # Store processed detections for this timestamp
            results[frame.time_ms] = processed_detections
            
        return results
        
    def fuse_detections(
        self,
        face_detections: List[Dict[str, Any]],
        object_detections: List[Dict[str, Any]],
        time_ms: int
    ) -> List[SalientRegion]:
        """
        Fuse face and object detections for a single frame.
        
        Args:
            face_detections: List of face detections
            object_detections: List of object detections
            time_ms: Timestamp in milliseconds
            
        Returns:
            List of fused salient regions
        """
        # Convert raw detections to SalientRegion objects
        salient_regions = []
        
        # Process face detections
        for i, face in enumerate(face_detections):
            signal_type = SignalType(standard=StandardSignalType.FACE_FULL)
            
            region = SalientRegion(
                x=face.get('x', 0),
                y=face.get('y', 0),
                width=face.get('width', 0),
                height=face.get('height', 0),
                score=face.get('confidence', 0),
                signal_type=signal_type,
                tracking_id=i  # Use index as tracking ID
            )
            salient_regions.append(region)
        
        # Process object detections
        for i, obj in enumerate(object_detections):
            obj_class = obj.get('class', '').lower()
            
            if obj_class == 'person':
                signal_type = SignalType(standard=StandardSignalType.HUMAN)
            elif obj_class in ['cat', 'dog']:
                signal_type = SignalType(standard=StandardSignalType.PET)
            elif obj_class in ['car', 'truck', 'bus']:
                signal_type = SignalType(standard=StandardSignalType.CAR)
            else:
                signal_type = SignalType(standard=StandardSignalType.OBJECT)
                
            region = SalientRegion(
                x=obj.get('x', 0),
                y=obj.get('y', 0),
                width=obj.get('width', 0),
                height=obj.get('height', 0),
                score=obj.get('confidence', 0),
                signal_type=signal_type,
                tracking_id=i + len(face_detections)  # Offset IDs to avoid collision with faces
            )
            salient_regions.append(region)
            
        # Add to internal buffer for scene processing
        detections_by_source = {
            0: salient_regions  # Using a single source for simplicity
        }
        self.add_frame(detections_by_source, time_ms)
        
        # Process immediately for single-frame mode
        if not self.process_by_scene:
            results = self.process_scene()
            if time_ms in results:
                return results[time_ms]
                
        return salient_regions 