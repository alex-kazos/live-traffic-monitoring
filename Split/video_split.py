import sys
import os
import ffmpeg


def split_video(input_path, segment_time=120, output_dir=None):
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    # Create the output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(input_path) or "."

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    # Put the segments precisely into the target directory
    output_pattern = os.path.join(output_dir, f"{file_name}_part_%03d.mp4")

    print(
        f"Splitting '{input_path}' into chunks of {segment_time} seconds -> {output_dir}"
    )

    try:
        (
            ffmpeg.input(input_path)
            .output(
                output_pattern,
                f="segment",
                segment_time=segment_time,
                reset_timestamps=1,
                vcodec="copy",
                acodec="copy",
            )
            .run()
        )
        print("Splitting completed successfully")

    except ffmpeg.Error as e:
        print("FFmpeg Error:")
        print(e.stderr.decode() if e.stderr else e)


if __name__ == "__main__":
    # Fallback to env variables if no sys.argv provided
    input_video = os.environ.get("INPUT_VIDEO")
    output_dir = os.environ.get("OUTPUT_DIR")
    segment_seconds = int(os.environ.get("SEGMENT_SECONDS", "120"))

    if len(sys.argv) == 2:
        input_video = sys.argv[1]

    if not input_video:
        print("Usage: python video_split.py <input_video>")
        print("Or set INPUT_VIDEO environment variable.")
        sys.exit(1)

    split_video(input_video, segment_time=segment_seconds, output_dir=output_dir)
