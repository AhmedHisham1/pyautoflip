import os
import tempfile
import aiohttp
from urllib.parse import urlparse

from cog import BasePredictor, Input, Path
from pyautoflip import reframe_video


class Predictor(BasePredictor):
    async def setup(self):
        """
        Load the model into memory to make running multiple predictions efficient.
        In this case, we don't need to load any model since pyautoflip handles it.
        """
        pass

    async def download_video(self, url: str) -> str:
        """
        Download video from URL to a temporary file.
        
        Args:
            url: URL to the video file
            
        Returns:
            Path to the downloaded video file
        """
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        
        # Get the filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename or extension found in URL, use a default
        if not filename or '.' not in filename:
            filename = "input_video.mp4"
            
        # Full path to save the video
        input_path = os.path.join(temp_dir, filename)
        
        # Download the video asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download video from {url}. Status code: {response.status}")
                
                with open(input_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                
        return input_path

    async def predict(
        self,
        video_url: str = Input(description="URL to the video file"),
        target_aspect_ratio: str = Input(description="Target aspect ratio as width:height (e.g., '9:16')"),
        padding_method: str = Input(
            description="Method for padding when content cannot be fully included",
            choices=["blur", "solid_color"],
            default="blur"
        ),
    ) -> Path:
        """
        Run AutoFlip video reframing.
        
        Args:
            video_url: URL to the video file to reframe
            target_aspect_ratio: Target aspect ratio as width:height (e.g., '9:16')
            padding_method: Method for padding when content cannot be fully included
            
        Returns:
            Path to the reframed video
        """
        # Create output directory
        temp_dir = tempfile.mkdtemp()
        output_filename = "reframed_video.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Download the video
        try:
            input_path = await self.download_video(video_url)
        except Exception as e:
            raise ValueError(f"Error downloading video: {str(e)}")
        
        # Process the video
        try:
            processed_video_path = reframe_video(
                input_path=input_path,
                output_path=output_path,
                target_aspect_ratio=target_aspect_ratio,
                padding_method=padding_method,
            )
            
            # Return the path as a Cog Path object
            return Path(processed_video_path)
            
        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")
