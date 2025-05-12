# Sleep Analysis Pipeline User Guide

The DLC analysed video and the dlc_results.csv file need to be in the input directory

## Running the pipeline:

### Single directoy use:
python sleep_analysis_pipeline.py /path/to/data_directory

### Batch processing with a text file:
python sleep_analysis_pipeline.py /path/to/directory_list.txt
Text file with directories, all processed with same parameters
/User/Experiment/projects/video_analysis/video_1
/User/Experiment/projects/video_analysis/video_2
/User/Experiment/projects/video_analysis/video_3

### Batch processing with custom parameters per directory using JSON file:
python sleep_analysis_pipeline.py /path/to/config.json


## Command-Line Parameters
All parameters have sensible defaults but can be customized:

Parameter	Default	Description
--frame-rate	60	Video frame rate in frames per second
--likelihood-threshold	0.95	Minimum likelihood value for DLC tracking points (0-1)
--speed-threshold	60.0	Speed threshold for sleep detection in pixels/second
--posture-threshold	60.0	Posture change threshold for sleep detection in pixels/second
--angular-threshold	50.0	Angular velocity threshold in degrees/second
--min-sleep	10.0	Minimum duration of a sleep bout in seconds
--no-plots	False	Disable generation of analysis plots
--pause	2	Seconds to pause between processing directories in batch mode
Can be used to customize analysis, e.g., use: python sleep_analysis_pipeline.py /path/to/data_directory --speed-threshold 40

## JSON config file:
{
  "/path/to/experiment1directory": {
    "frame_rate": 60,
    "speed_threshold": 40.0,
    "min_sleep_duration_seconds": 5.0 #remaining parameters will use default
  },
  "/path/to/experiment2directory": {
    "frame_rate": 60,
    "posture_threshold": 50.0,
    "angular_threshold": 45.0
  },
  "/path/to/experiment3directory": {} #default parameters will be used
}
