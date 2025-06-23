from typing import List

from scenedetect import detect, ContentDetector


class ShotBoundaryDetector:
    """
    Detector for identifying shot boundaries (scene changes) in videos using PySceneDetect.
    """

    def __init__(
        self,
        threshold: float = 27.0,  # ContentDetector threshold
        min_scene_length: int = 15,  # Minimum scene length in frames
    ):
        """
        Initialize the shot boundary detector.

        Args:
            threshold: Threshold for content detection (higher means less sensitive)
            min_scene_length: Minimum scene length in frames
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length

    def detect(self, video_path: str) -> List[int]:
        """
        Detect shot boundaries directly from a video file using SceneDetect.

        Args:
            video_path: Path to the video file

        Returns:
            List of frame indices where shot boundaries occur
        """
        # Use the new non-deprecated API for reduced memory usage
        scene_list = detect(
            video_path,
            ContentDetector(
                threshold=self.threshold, min_scene_len=self.min_scene_length
            ),
        )

        # Convert scene list to boundary frames
        shot_boundaries = []
        for scene in scene_list:
            # Start frame of each scene (except the first one) is a boundary
            if scene[0].frame_num > 0:  # Skip the first scene's start (which is 0)
                shot_boundaries.append(scene[0].frame_num)

        return shot_boundaries
