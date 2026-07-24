from typing import List, Optional

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

    def detect(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> List[int]:
        """
        Detect shot boundaries directly from a video file using SceneDetect.

        Args:
            video_path: Path to the video file
            start_frame: Only scan from this frame (absolute) onward
            end_frame: Stop scanning at this frame (absolute)

        Returns:
            List of absolute frame indices where shot boundaries occur.
            Scanning a subrange decodes only that range — analyzing a short
            segment of a long video costs seconds instead of minutes.
        """
        kwargs = {}
        if start_frame is not None:
            kwargs["start_time"] = int(start_frame)
        if end_frame is not None:
            kwargs["end_time"] = int(end_frame)

        # Use the new non-deprecated API for reduced memory usage
        scene_list = detect(
            video_path,
            ContentDetector(
                threshold=self.threshold, min_scene_len=self.min_scene_length
            ),
            **kwargs,
        )

        # Scene starts are boundaries — except the first scene's, which is
        # just where the scan began (frame 0, or start_frame for subranges)
        return [scene[0].frame_num for scene in scene_list[1:]]
