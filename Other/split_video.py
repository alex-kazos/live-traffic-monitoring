import os
import gdown
from moviepy import VideoFileClip
import math

### CONFIGURATION
DRIVE_LINK = "https://drive.google.com/file/d/1igBVolFaFbOaQvGtsOhxWf1Y4ESQMgh7/view?usp=drive_link"
DOWNLOADED_VIDEO_NAME = "Road traffic video for object recognition.mp4"
SPLIT_TIME_SECONDS = 10

"""
Note for future self: 

Paths could be handled with pathlib, but for simplicity we'll just use os.path.join here. 
Unless someone wants to refactor it later.
"""

def download_from_drive(drive_url:str, output_filename:str):
    """Downloads a file from Google Drive using a shareable link.

    Parameters
    ----------
    drive_url : str
        The shareable link to the Google Drive file.
    output_filename : str
        The name to save the downloaded file as (including extension).
    """

    print(f"Downloading video from Google Drive...")

    # Create a directory to store the downloaded file
    if not os.path.exists("../Downloads"):
        os.mkdir("../Downloads")

    output_path = os.path.join("../Downloads", output_filename)

    try:
        # Extract the unique file ID from the full URL
        file_id = drive_url.split('/d/')[1].split('/')[0]

        # Tell gdown to use the ID directly to bypass HTML warning pages
        downloaded_file = gdown.download(id=file_id, output=output_path, quiet=False)

        # Chek Video file is valid
        if downloaded_file is None:
            raise Exception("Failed to download the video.")
        return downloaded_file

    except Exception as e:
        print(f"Error downloading file: {e}")
        raise



def split_video(video_name:str, segment_length_seconds:int=120):
    """Splits a video into smaller segments.

    Parameters
    ----------
    video_name : str
        The name of the video file to Split (including extension).
    segment_length_seconds : int
        The length of each segment in seconds. Default is 120 seconds (2 minutes).

    """
    # Configure paths
    video_path = os.path.join("../Downloads", video_name)

    print(f"Loading video: {video_path}")
    video = VideoFileClip(video_path)
    total_duration = video.duration

    # Calculate how many segments we will have
    total_segments = math.ceil(total_duration / segment_length_seconds)
    print(f"Total video duration: {total_duration:.2f} seconds.") # fixed on 2 mins ( maybe more or less depending on the video )
    print(f"Splitting into {total_segments} segment(s) of ~{segment_length_seconds} seconds each.")

    # Get the base name without extension for naming the output files
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a directory to store the Segments files
    if not os.path.exists("../Downloads/Segments"):
        os.mkdir("../Downloads/Segments")

    # Ensure the output directory `Downloads/Segments` exists
    segments_dir = os.path.join("../Downloads", "Segments")

    for i in range(total_segments):
        if i <= 2:
            start_time = i * segment_length_seconds
            # Make sure the end time doesn't exceed the total duration
            end_time = min((i + 1) * segment_length_seconds, total_duration)

            output_name = f"{base_name}_part_{i + 1}.mp4"
            output_path = os.path.join(segments_dir, output_name)
            print(f"Exporting {output_path} (From {start_time}s to {end_time}s)...")

            # Create a subclip and write it to a file
            clip = video.subclipped(start_time, end_time)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        else:
            break
    # Close the video file to free up system resources
    video.close()
    print("Video splitting complete!")


if __name__ == "__main__":
    # Download the video
    download_from_drive(DRIVE_LINK, DOWNLOADED_VIDEO_NAME)

    # Split the video
    # TO-DO: This is slow as shit. Add ThreadPoolExecutor
    split_video(DOWNLOADED_VIDEO_NAME, SPLIT_TIME_SECONDS)

    # Remove the for cleanup
    # temp_vide_path = os.path.join("Downloads", DOWNLOADED_VIDEO_NAME)
    # os.remove(temp_vide_path)