import os
import subprocess as sp

def subset_video(ffmpeg_path, input_video_path, output_directory, subset_start_point=0, subset_duration=None):
    """
    Converts a subset of an MJ2 video to MP4 format.

    Parameters
    ----------
    ffmpeg_path : str
        Path to the ffmpeg executable (can be just 'ffmpeg' if in PATH)
    input_video_path : str
        Full path to the input MJ2 video
    output_directory : str
        Folder where the output MP4 should be saved
    subset_start_point : int or float
        Time in seconds to start the subset
    subset_duration : int or float
        Duration of the subset in seconds

    Returns
    -------
    subset_video_path : str
        Full path to the saved MP4 video
    subset_video_name : str
        Filename of the saved MP4 video
    """

    input_video_name = os.path.basename(input_video_path)
    input_name_base = os.path.splitext(input_video_name)[0]

    # Create a unique suffix based on parameters
    start_str = f"start{int(subset_start_point)}"
    dur_str = f"dur{int(subset_duration)}" if subset_duration is not None else "full"
    subset_video_name = f"{input_name_base}_subset_{start_str}_{dur_str}.mp4"
    subset_video_path = os.path.join(output_directory, subset_video_name)

    ffmpeg_args = [
        ffmpeg_path,
        '-ss', str(subset_start_point),
        '-i', input_video_path
    ]

    if subset_duration is not None:
        ffmpeg_args += ['-t', str(subset_duration)]

    ffmpeg_args += [
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-y',
        subset_video_path
    ]

    print("Running ffmpeg command:", ' '.join(ffmpeg_args))
    sp.call(ffmpeg_args)

    return subset_video_path, subset_video_name



if __name__ == "__main__":
    ffmpeg_exec = 'ffmpeg'
    mj2_file = r"\\zaru.cortexlab.net\Subjects\AV049\2023-08-08\2\2023-08-08_2_AV049_topCam.mj2"
    output_dir = r"C:\Users\Experiment\Projects\video_conversions\subjects\AV049\2023-08-08_2"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If desired, add a subset duration (in seconds) to the conversion. Otherwise, whole video will be converted.
    output_path, output_name = subset_video(ffmpeg_exec, mj2_file, output_dir, 0)
    print("Converted video saved at:", output_path)
