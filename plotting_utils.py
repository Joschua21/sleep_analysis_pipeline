import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
import colorsys
import matplotlib.patches as mpatches

try:
    import cv2
    import moviepy.editor as mpy
    VIDEO_LIBS_AVAILABLE = True
except ImportError:
    VIDEO_LIBS_AVAILABLE = False
    print("Warning: moviepy or opencv-python not installed. Video generation functionality will be disabled.")
    print("Install them using: pip install moviepy opencv-python")



def interpolate_gaps_conditionally(series: pd.Series, max_gap_length: int) -> pd.Series:
    """
    Interpolates NaN values in a Series using linear interpolation, but only
    if the consecutive NaN gap is shorter than or equal to max_gap_length.

    Args:
        series (pd.Series): The input Series with potential NaNs.
        max_gap_length (int): The maximum length of a NaN gap to interpolate.

    Returns:
        pd.Series: Series with short NaN gaps interpolated.
    """
    s_out = series.copy()
    is_na = s_out.isna()

    if not is_na.any(): # No NaNs, nothing to do
        return s_out

    # Identify groups of consecutive NaNs
    na_group_ids = is_na.ne(is_na.shift()).cumsum()

    # Iterate over each NaN block
    # We are interested in the groups that are True (i.e., are NaN blocks)
    for group_id, na_block_series in s_out[is_na].groupby(na_group_ids[is_na]):
        if not na_block_series.empty and len(na_block_series) <= max_gap_length:
            # This block of NaNs is short enough to interpolate.
            # We perform a full interpolation on the original series (s_out)
            # to get the correct values based on surrounding non-NaNs,
            # and then apply these interpolated values only to the current short gap.
            
            # Temporarily interpolate the whole series to get potential values
            temp_interpolated_series = s_out.interpolate(method='linear', limit_direction='both')
            
            # Copy values from temp_interpolated_series to s_out *only* for this short gap's indices
            s_out.loc[na_block_series.index] = temp_interpolated_series.loc[na_block_series.index]
            
    return s_out

# Add smoothing_window_seconds parameter
def plot_speed(df_dlc, df_displacements, final_bodyparts_list, frame_rate, output_dir, base_filename, plot_individual=True, save_plot=True, smoothing_window_seconds=1.0):
    """
    Plots the average speed and optionally the speed of individual bodyparts over time.
    Can apply a rolling window average to smooth the average speed line.

    Args:
        # ... other args ...
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to True.
        smoothing_window_seconds (float, optional): Duration of the rolling window for smoothing
                                                    the average speed plot in seconds.
                                                    Set to 0 or None to disable smoothing. Defaults to 1.0.
    """
    print(f"\nGenerating speed plot... Individual parts: {plot_individual}, Smoothing: {smoothing_window_seconds}s")
    fig, ax = plt.subplots(figsize=(8, 3))
    # Calculate time axis
    time_seconds = df_dlc.index / frame_rate

    # Plot Average Speed (potentially smoothed)
    if ('analysis', 'speed_pixels_per_second') in df_dlc.columns:
        average_speed = df_dlc[('analysis', 'speed_pixels_per_second')]
        plot_label = 'Average Speed'

        # Apply smoothing if requested
        if smoothing_window_seconds and smoothing_window_seconds > 0:
            # Calculate window size in frames (must be an integer >= 1)
            window_size = max(1, int(smoothing_window_seconds * frame_rate))
            # Apply rolling mean - center=True places the window centered on the point
            # min_periods=1 ensures calculation even if window is not full (e.g., at edges)
            average_speed_smoothed = average_speed.rolling(window=window_size, center=True, min_periods=1).mean()
            plot_data = average_speed_smoothed
            plot_label = f'Average Speed ({smoothing_window_seconds}s Smoothed)'
            print(f"  Applying rolling average with window size: {window_size} frames ({smoothing_window_seconds}s)")
        else:
            plot_data = average_speed # Plot original if no smoothing

        ax.plot(time_seconds, plot_data, label=plot_label, linewidth=2, color='black', zorder=10)
    else:
        print("Warning: Average speed column not found for plotting.")

    # Plot Individual Bodypart Speeds (Optional - not smoothed)
    if plot_individual:
        for bp in final_bodyparts_list:
            displacement_col = f'{bp}_displacement'
            if displacement_col in df_displacements.columns:
                individual_speed = df_displacements[displacement_col] * frame_rate
                ax.plot(time_seconds, individual_speed, label=f'{bp} Speed', alpha=0.6)
            else:
                 print(f"Warning: Displacement column '{displacement_col}' not found for plotting individual speed.")

    # Customize plot
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speed (pixels/second)')
    ax.set_title('Mouse Speed Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save plot
    if save_plot:
        # Add suffix if smoothed
        smooth_suffix = f'_smoothed{smoothing_window_seconds}s' if smoothing_window_seconds and smoothing_window_seconds > 0 else ''
        plot_filename = os.path.join(output_dir, base_filename + '_speed_plot' + smooth_suffix + '.png')
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Speed plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    return fig


def create_synced_video_with_plot(
    video_path: str,
    speed_data: pd.Series,
    frame_rate: float,
    output_video_path: str,
    median_coords: pd.DataFrame = None, # New parameter for median coordinates
    plot_width_seconds: float = 5.0,
    plot_height_pixels: int = 200,
    median_point_radius: int = 5, # Radius of the median point
    median_point_color: tuple = (0, 0, 0) # Black color for the point (BGR for OpenCV)
):
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting synchronized video creation with median point overlay: {output_video_path}")
    original_backend = matplotlib.get_backend()
    print(f"Original matplotlib backend: {original_backend}")
    matplotlib.use('Agg') # Switch to a non-interactive backend for performance
    print(f"Temporarily switched matplotlib backend to: Agg")

    fig = None # Initialize fig to None for the finally block
    video_clip_orig = None

    try:
        video_clip_orig = mpy.VideoFileClip(video_path)
        w, h = video_clip_orig.w, video_clip_orig.h
        vid_duration = video_clip_orig.duration
        vid_fps = video_clip_orig.fps

        if abs(vid_fps - frame_rate) > 1:
            print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
            actual_frame_rate = vid_fps
        else:
            actual_frame_rate = frame_rate

        # --- Prepare median coordinates if provided ---
        median_x_np = None
        median_y_np = None
        if median_coords is not None and not median_coords.empty:
            if ('analysis', 'median_x') in median_coords.columns and \
               ('analysis', 'median_y') in median_coords.columns:
                median_x_np = median_coords[('analysis', 'median_x')].to_numpy()
                median_y_np = median_coords[('analysis', 'median_y')].to_numpy()
                print("Median coordinates provided for overlay.")
            else:
                print("Warning: Median coordinates DataFrame provided but 'median_x' or 'median_y' columns are missing. No overlay will be drawn.")
        else:
            print("No median coordinates provided for overlay.")


        # --- Function to draw median point on each frame ---
        def draw_median_on_frame(get_frame, t):
            frame_orig = get_frame(t) # Get the original frame
            frame = frame_orig.copy() # <--- MAKE A WRITABLE COPY
            current_frame_idx = int(t * actual_frame_rate)

            if median_x_np is not None and median_y_np is not None and \
               current_frame_idx < len(median_x_np) and current_frame_idx < len(median_y_np):
                
                mx = median_x_np[current_frame_idx]
                my = median_y_np[current_frame_idx]

                if not np.isnan(mx) and not np.isnan(my):
                    center_coordinates = (int(mx), int(my))
                    # Draw the circle. Note: frame is a NumPy array.
                    # OpenCV uses BGR color format by default.
                    try:
                        cv2.circle(frame, center_coordinates, median_point_radius, median_point_color, -1) # -1 for filled circle
                    except Exception as e_cv2:
                        print(f"Error drawing circle at frame {current_frame_idx} time {t}: {e_cv2}")
            return frame

        # Apply the drawing function to the video clip
        video_clip_processed = video_clip_orig.fl(draw_median_on_frame)
        # --- End Median Point Drawing ---


        speed_data_np = speed_data.fillna(0).to_numpy()
        valid_speeds = speed_data_np[np.isfinite(speed_data_np)]

        if len(valid_speeds) > 0:
            max_s = np.max(valid_speeds) * 1.1 
        else:
            max_s = 0 # Fallback if no valid speed data
        
        if max_s == 0 or np.isnan(max_s) or not np.isfinite(max_s): # Ensure max_speed is a positive finite number
            max_speed_for_plot = 100.0 # Default y-limit if max_s is problematic
        else:
            max_speed_for_plot = max_s

        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        fig, ax = plt.subplots(figsize=(w / 80, plot_height_pixels / 80), dpi=80)
        line, = ax.plot([], [], color='r')
        vline = ax.axvline(0, color='lime', linestyle='--', lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (px/s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_speed_for_plot)
        fig.tight_layout(pad=0.5)

        def make_plot_frame(t):
            # ... (rest of make_plot_frame remains the same as your optimized version)
            current_frame = int(t * actual_frame_rate)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = current_frame + 1
            plot_data_segment = speed_data_np[start_frame_idx:end_frame_idx]
            
            # Ensure time_axis_segment aligns with plot_data_segment
            # It should represent the actual time values for the x-axis of the plot window
            time_axis_plot_window_data = np.arange(start_frame_idx, end_frame_idx) * time_per_frame
            
            line.set_data(time_axis_plot_window_data, plot_data_segment)
            
            # Current time marker position (this 't' is the video's current time)
            vline.set_xdata([t, t])
            
            # Set x-axis limits for the scrolling window effect
            # The window shows `plot_width_seconds` of data, with the current time 't' ideally at the right edge or slightly before.
            plot_window_end_time = t + (time_per_frame * 5) # Show a little bit past current time for context
            plot_window_start_time = max(0.0, plot_window_end_time - plot_width_seconds)

            # Adjust if current time t is less than the plot_width_seconds
            if t < plot_width_seconds - (time_per_frame * 5) : # Ensure the window starts at 0 if t is too small
                plot_window_start_time = 0.0
                plot_window_end_time = plot_width_seconds
            
            ax.set_xlim(plot_window_start_time, plot_window_end_time)
            
            fig.canvas.draw() # This is the expensive call
            img_buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(img_buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            return img[:, :, :3]

        plot_clip = mpy.VideoClip(make_plot_frame, duration=vid_duration)
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels))

        # Use the processed video clip (with median point) here
        final_clip = mpy.clips_array([[video_clip_processed], [plot_clip]])

        print(f"Writing final video to {output_video_path}...")
        # ... (write_videofile call remains the same, consider the h264_nvenc option)
        try:
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='h264_nvenc', audio=False,
                threads=4, preset='fast', logger='bar'
            )
            print("Video encoding with h264_nvenc successful.")
        except Exception as e_nvenc:
            print(f"Warning: h264_nvenc encoding failed ({e_nvenc}). Falling back to libx264.")
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='libx264', audio=False,
                threads=4, preset='ultrafast', logger='bar'
            )

        plt.close(fig)
        video_clip_orig.close() # Close the original video clip
        # video_clip_processed doesn't need explicit close if it's just a result of .fl()
        print("Video creation complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        if 'video_clip_orig' in locals() and hasattr(video_clip_orig, 'close'): video_clip_orig.close()


# Video conversion for adding sleep bout periods and changes in arousal

def create_synced_video_with_sleep_analysis(
    video_path: str,
    speed_data: pd.Series,
    frame_rate: float,
    output_video_path: str,
    df_sleep_bouts: pd.DataFrame,  # New parameter for sleep bout information
    median_coords: pd.DataFrame = None,
    plot_width_seconds: float = 5.0,
    plot_height_pixels: int = 200,
    median_point_radius: int = 5,
    median_point_color: tuple = (0, 0, 255),  # Red color (BGR for OpenCV)
    sleep_threshold: float = 75.0,  # Speed threshold for sleep
    arousal_low_threshold: float = 30.0,  # Lower threshold for arousal
    arousal_high_threshold: float = 40.0  # Higher threshold for arousal
):
    """
    Create a synchronized video with sleep analysis visualization.
    - Marks sleep periods (speed < sleep_threshold) with light green background
    - Changes line color based on arousal state:
      * Blue during sleep periods (< arousal_low_threshold)
      * Yellow-blue gradient when between arousal_low_threshold and arousal_high_threshold
      * Red-yellow gradient when between arousal_high_threshold and sleep_threshold
      * Black when not in sleep periods
    """
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting synchronized video creation with sleep analysis: {output_video_path}")
    original_backend = matplotlib.get_backend()
    print(f"Original matplotlib backend: {original_backend}")
    matplotlib.use('Agg')  # Switch to a non-interactive backend for performance
    print(f"Temporarily switched matplotlib backend to: Agg")

    fig = None  # Initialize fig to None for the finally block
    video_clip_orig = None

    try:
        video_clip_orig = mpy.VideoFileClip(video_path)
        w, h = video_clip_orig.w, video_clip_orig.h
        vid_duration = video_clip_orig.duration
        vid_fps = video_clip_orig.fps

        if abs(vid_fps - frame_rate) > 1:
            print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
            actual_frame_rate = vid_fps
        else:
            actual_frame_rate = frame_rate

        # --- Prepare median coordinates if provided ---
        median_x_np = None
        median_y_np = None
        if median_coords is not None and not median_coords.empty:
            if ('analysis', 'median_x') in median_coords.columns and \
               ('analysis', 'median_y') in median_coords.columns:
                median_x_np = median_coords[('analysis', 'median_x')].to_numpy()
                median_y_np = median_coords[('analysis', 'median_y')].to_numpy()
                print("Median coordinates provided for overlay.")
            else:
                print("Warning: Median coordinates DataFrame provided but 'median_x' or 'median_y' columns are missing. No overlay will be drawn.")
        else:
            print("No median coordinates provided for overlay.")

        # --- Prepare sleep bouts data ---
        sleep_periods = []
        if df_sleep_bouts is not None and not df_sleep_bouts.empty:
            for _, bout in df_sleep_bouts.iterrows():
                sleep_periods.append((bout['start_time_s'], bout['end_time_s']))
            print(f"Added {len(sleep_periods)} sleep periods for visualization.")
        else:
            print("No sleep periods data provided.")

        # --- Function to draw median point on each frame ---
        def draw_median_on_frame(get_frame, t):
            frame_orig = get_frame(t)  # Get the original frame
            frame = frame_orig.copy()  # Make a writable copy
            current_frame_idx = int(t * actual_frame_rate)

            if median_x_np is not None and median_y_np is not None and \
               current_frame_idx < len(median_x_np) and current_frame_idx < len(median_y_np):
                
                mx = median_x_np[current_frame_idx]
                my = median_y_np[current_frame_idx]

                if not np.isnan(mx) and not np.isnan(my):
                    center_coordinates = (int(mx), int(my))
                    # Draw the circle
                    try:
                        cv2.circle(frame, center_coordinates, median_point_radius, median_point_color, -1)
                    except Exception as e_cv2:
                        print(f"Error drawing circle at frame {current_frame_idx} time {t}: {e_cv2}")
            return frame

        # Apply the drawing function to the video clip
        video_clip_processed = video_clip_orig.fl(draw_median_on_frame)

        speed_data_np = speed_data.fillna(0).to_numpy()
        valid_speeds = speed_data_np[np.isfinite(speed_data_np)]

        if len(valid_speeds) > 0:
            max_s = np.max(valid_speeds) * 1.1 
        else:
            max_s = 0  # Fallback if no valid speed data
        
        if max_s == 0 or np.isnan(max_s) or not np.isfinite(max_s):
            max_speed_for_plot = 100.0  # Default y-limit if max_s is problematic
        else:
            max_speed_for_plot = max_s

        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        fig, ax = plt.subplots(figsize=(w / 80, plot_height_pixels / 80), dpi=80)
        line, = ax.plot([], [], color='k')  # Start with black line
        vline = ax.axvline(0, color='lime', linestyle='--', lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (px/s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_speed_for_plot)
        
        # Add threshold lines
        ax.axhline(sleep_threshold, color='r', linestyle='--', lw=0.5, alpha=0.5)
        ax.axhline(arousal_high_threshold, color='orange', linestyle=':', lw=0.5, alpha=0.5)
        ax.axhline(arousal_low_threshold, color='y', linestyle=':', lw=0.5, alpha=0.5)
        
        fig.tight_layout(pad=0.5)

        # Helper function to check if time is in sleep period
        def is_in_sleep_period(t):
            for start, end in sleep_periods:
                if start <= t < end:
                    return True
            return False
        
        # Helper function to determine color based on speed and sleep state
        def get_line_color(t, speed_value):
            if not is_in_sleep_period(t):
                return 'black'
            
            # In sleep period
            if speed_value < arousal_low_threshold:
                return 'blue'  # Base sleep color
            elif speed_value < arousal_high_threshold:
                # Gradient from blue to yellow based on position between thresholds
                ratio = (speed_value - arousal_low_threshold) / (arousal_high_threshold - arousal_low_threshold)
                # Mix blue (0,0,1) and yellow (1,1,0)
                r = ratio
                g = ratio
                b = 1 - ratio
                return (r, g, b)
            elif speed_value < sleep_threshold:
                # Gradient from yellow to red based on position between thresholds
                ratio = (speed_value - arousal_high_threshold) / (sleep_threshold - arousal_high_threshold)
                # Mix yellow (1,1,0) and red (1,0,0)
                r = 1.0
                g = 1.0 - ratio
                b = 0
                return (r, g, b)
            else:
                return 'black'  # Default case
        
        # Spans for sleep periods (created once)
        sleep_spans = []
        for start, end in sleep_periods:
            span = ax.axvspan(start, end, color='palegreen', alpha=0.3, zorder=0)
            sleep_spans.append(span)

        def make_plot_frame(t):
            current_frame = int(t * actual_frame_rate)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = current_frame + 1
            plot_data_segment = speed_data_np[start_frame_idx:end_frame_idx]
            
            time_axis_plot_window_data = np.arange(start_frame_idx, end_frame_idx) * time_per_frame
            
            # Determine if current time is in sleep period for coloring
            in_sleep = is_in_sleep_period(t)
            
            # Set up plot window limits
            plot_window_end_time = t + (time_per_frame * 5)
            plot_window_start_time = max(0.0, plot_window_end_time - plot_width_seconds)
            
            # Adjust if current time t is less than plot_width_seconds
            if t < plot_width_seconds - (time_per_frame * 5):
                plot_window_start_time = 0.0
                plot_window_end_time = plot_width_seconds
            
            ax.set_xlim(plot_window_start_time, plot_window_end_time)
            
            # Set line data
            line.set_data(time_axis_plot_window_data, plot_data_segment)
            
            # Update line color based on current speed and sleep state
            current_speed = speed_data_np[current_frame] if current_frame < len(speed_data_np) else 0
            line.set_color(get_line_color(t, current_speed))
            
            # Update current time marker
            vline.set_xdata([t, t])
            
            fig.canvas.draw()
            img_buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(img_buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            return img[:, :, :3]

        plot_clip = mpy.VideoClip(make_plot_frame, duration=vid_duration)
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels))

        # Use the processed video clip (with median point)
        final_clip = mpy.clips_array([[video_clip_processed], [plot_clip]])

        print(f"Writing final video to {output_video_path}...")
        try:
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='h264_nvenc', audio=False,
                threads=4, preset='fast', logger='bar'
            )
            print("Video encoding with h264_nvenc successful.")
        except Exception as e_nvenc:
            print(f"Warning: h264_nvenc encoding failed ({e_nvenc}). Falling back to libx264.")
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='libx264', audio=False,
                threads=4, preset='ultrafast', logger='bar'
            )

        plt.close(fig)
        video_clip_orig.close()
        print("Video creation with sleep analysis complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        if 'video_clip_orig' in locals() and hasattr(video_clip_orig, 'close'):
            video_clip_orig.close()
    finally:
        matplotlib.use(original_backend)  # Restore the original matplotlib backend
        print(f"Restored matplotlib backend to: {original_backend}")

def plot_body_posture_metric(
    df_dlc,
    metric_column_tuple, # e.g., ('analysis', 'avg_dist_to_median')
    frame_rate,
    output_dir_path,     # Renamed from output_dir for consistency
    base_output_name,    # Renamed from base_filename for consistency
    save_plot=True,
    display_plot=True,
    smoothing_window_seconds_metric=0.25, # Default smoothing for the new metric
    plot_with_speed=True,
    speed_column_tuple=('analysis', 'speed_pixels_per_second'),
    smoothing_window_seconds_speed=0.25  # Default smoothing for speed on this plot
):
    """
    Plots a calculated body posture metric (e.g., average distance of bodyparts to median)
    over time, optionally with speed on a secondary axis.

    Args:
        df_dlc (pd.DataFrame): DataFrame containing the analysis data.
        metric_column_tuple (tuple): Tuple identifying the metric column, e.g., ('analysis', 'avg_dist_to_median').
        frame_rate (float): Video frame rate in FPS.
        output_dir_path (str): Path to the directory where the plot will be saved.
        base_output_name (str): Base name for the output plot file.
        save_plot (bool): Whether to save the plot.
        display_plot (bool): Whether to display the plot.
        smoothing_window_seconds_metric (float): Smoothing window in seconds for the posture metric. 0 for no smoothing.
        plot_with_speed (bool): Whether to plot speed on a secondary y-axis.
        speed_column_tuple (tuple): Tuple identifying the speed column if plotting with speed.
        smoothing_window_seconds_speed (float): Smoothing window in seconds for the speed trace. 0 for no smoothing.
    """
    if metric_column_tuple not in df_dlc.columns:
        print(f"Error: Metric column {metric_column_tuple} not found in DataFrame.")
        if display_plot: # Avoids error if plt is not meant to be shown
             plt.close(plt.gcf()) if plt.get_fignums() else None
        return

    fig, ax1 = plt.subplots(figsize=(15, 6))
    time_axis_seconds = df_dlc.index / frame_rate

    # --- Plot Body Posture Metric ---
    metric_data = df_dlc[metric_column_tuple].copy()
    if smoothing_window_seconds_metric > 0 and frame_rate > 0:
        smoothing_window_frames = int(smoothing_window_seconds_metric * frame_rate)
        if smoothing_window_frames < 1:
            smoothing_window_frames = 1
        metric_data_smoothed = metric_data.rolling(window=smoothing_window_frames, min_periods=1, center=True).mean()
        label_metric = f'Smoothed {metric_column_tuple[-1]} ({smoothing_window_seconds_metric}s window)'
    else:
        metric_data_smoothed = metric_data
        label_metric = f'{metric_column_tuple[-1]} (px)'

    color_metric = 'tab:red'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(f'{metric_column_tuple[-1]} (pixels)', color=color_metric)
    ax1.plot(time_axis_seconds, metric_data_smoothed, color=color_metric, label=label_metric, lw=1.5)
    ax1.tick_params(axis='y', labelcolor=color_metric)
    ax1.grid(True, linestyle=':', alpha=0.6)

    lines, labels = ax1.get_legend_handles_labels()

    if plot_with_speed and speed_column_tuple in df_dlc.columns:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        speed_data = df_dlc[speed_column_tuple].copy()

        if smoothing_window_seconds_speed > 0 and frame_rate > 0:
            smoothing_window_frames_speed = int(smoothing_window_seconds_speed * frame_rate)
            if smoothing_window_frames_speed < 1:
                smoothing_window_frames_speed = 1
            speed_data_smoothed = speed_data.rolling(window=smoothing_window_frames_speed, min_periods=1, center=True).mean()
            label_speed = f'Smoothed Speed ({smoothing_window_seconds_speed}s window)'
        else:
            speed_data_smoothed = speed_data
            label_speed = 'Speed (px/s)'

        color_speed = 'tab:blue'
        ax2.set_ylabel('Speed (pixels/second)', color=color_speed)
        ax2.plot(time_axis_seconds, speed_data_smoothed, color=color_speed, label=label_speed, lw=1, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_speed)
        
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    
    fig.suptitle('Body Posture Metric and Speed Over Time', fontsize=14) # Changed title to suptitle
    ax1.legend(lines, labels, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle


    if save_plot:
        plot_filename = os.path.join(output_dir_path, base_output_name + '_body_posture_metric.png')
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"Body posture metric plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving body posture metric plot: {e}")

    if display_plot:
        plt.show()
    else:
        plt.close(fig)


def create_comprehensive_sleep_analysis_video(
    video_path: str,
    speed_data: pd.Series, 
    body_movement_derivative: pd.Series,
    angular_velocity_data: pd.Series,
    smoothed_angle_data: pd.Series,
    frame_rate: float,
    output_video_path: str,
    df_midpoints_pca_raw: pd.DataFrame = None,
    df_speed_sleep_bouts: pd.DataFrame = None,
    df_posture_sleep_bouts: pd.DataFrame = None,
    df_angular_sleep_bouts: pd.DataFrame = None,
    bodypart_coordinate_sets: dict = None,
    median_coords: pd.DataFrame = None,
    plot_width_seconds: float = 5.0,
    plot_height_pixels: int = 100,  # Reduced from 150 to 100 (2/3 of original)
    median_point_radius: int = 5,
    median_point_color: tuple = (0, 0, 255),
    arrow_color: tuple = (130, 0, 130),
    arrow_length: int = 40,
    arrow_size: int = 10,
    arrow_smoothing_frames: int = 15,
    speed_threshold: float = 60.0,
    posture_threshold: float = 60.0,
    angular_threshold: float = 50.0
):
    """
    Create a synchronized video with comprehensive sleep analysis visualization, with optimizations:
    - Pre-renders static elements that don't change across frames
    - Removes grid lines for performance 
    - Uses 2/3 the original plot height for better performance
    - Uses efficient drawing with fewer redraws per frame
    """
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting optimized comprehensive sleep analysis video creation: {output_video_path}")
    original_backend = matplotlib.get_backend()
    print(f"Original matplotlib backend: {original_backend}")
    matplotlib.use('Agg')
    print(f"Temporarily switched matplotlib backend to: Agg")

    fig = None
    video_clip_orig = None
    polar_fig = None

    try:
        # Load the video
        video_clip_orig = mpy.VideoFileClip(video_path)
        w, h = video_clip_orig.w, video_clip_orig.h
        vid_duration = video_clip_orig.duration
        vid_fps = video_clip_orig.fps

        if abs(vid_fps - frame_rate) > 1:
            print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
            actual_frame_rate = vid_fps
        else:
            actual_frame_rate = frame_rate

        # --- Prepare PCA data for arrow visualization ---
        have_pca_data = False
        pca_dx_np = None
        pca_dy_np = None
        pca_mean_x_np = None
        pca_mean_y_np = None
        
        if df_midpoints_pca_raw is not None and not df_midpoints_pca_raw.empty:
            if all(col in df_midpoints_pca_raw.columns for col in ['pca_dx', 'pca_dy', 'pca_mean_x', 'pca_mean_y']):
                pca_dx_np = df_midpoints_pca_raw['pca_dx'].to_numpy()
                pca_dy_np = df_midpoints_pca_raw['pca_dy'].to_numpy()
                pca_mean_x_np = df_midpoints_pca_raw['pca_mean_x'].to_numpy()
                pca_mean_y_np = df_midpoints_pca_raw['pca_mean_y'].to_numpy()
                have_pca_data = True
                print("PCA data provided for arrow visualization.")
                
                # Pre-calculate smoothed PCA data for arrow
                smoothed_pca_dx = np.zeros_like(pca_dx_np)
                smoothed_pca_dy = np.zeros_like(pca_dy_np)
                smoothed_pca_mean_x = np.zeros_like(pca_mean_x_np)
                smoothed_pca_mean_y = np.zeros_like(pca_mean_y_np)
                
                # Apply rolling average for smoothing
                for i in range(len(pca_dx_np)):
                    start_idx = max(0, i - arrow_smoothing_frames // 2)
                    end_idx = min(len(pca_dx_np), i + arrow_smoothing_frames // 2 + 1)
                    
                    # Handle NaN values
                    dx_slice = pca_dx_np[start_idx:end_idx]
                    dy_slice = pca_dy_np[start_idx:end_idx]
                    mean_x_slice = pca_mean_x_np[start_idx:end_idx]
                    mean_y_slice = pca_mean_y_np[start_idx:end_idx]
                    
                    valid_indices = ~np.isnan(dx_slice) & ~np.isnan(dy_slice) & ~np.isnan(mean_x_slice) & ~np.isnan(mean_y_slice)
                    
                    if np.any(valid_indices):
                        smoothed_pca_dx[i] = np.mean(dx_slice[valid_indices])
                        smoothed_pca_dy[i] = np.mean(dy_slice[valid_indices])
                        smoothed_pca_mean_x[i] = np.mean(mean_x_slice[valid_indices])
                        smoothed_pca_mean_y[i] = np.mean(mean_y_slice[valid_indices])
                    else:
                        smoothed_pca_dx[i] = np.nan
                        smoothed_pca_dy[i] = np.nan
                        smoothed_pca_mean_x[i] = np.nan
                        smoothed_pca_mean_y[i] = np.nan
                
                # Normalize all direction vectors after smoothing
                norm = np.sqrt(smoothed_pca_dx**2 + smoothed_pca_dy**2)
                valid_norm = ~np.isnan(norm) & (norm > 0)
                if np.any(valid_norm):
                    smoothed_pca_dx[valid_norm] = smoothed_pca_dx[valid_norm] / norm[valid_norm]
                    smoothed_pca_dy[valid_norm] = smoothed_pca_dy[valid_norm] / norm[valid_norm]
                
                # Replace original arrays with smoothed versions
                pca_dx_np = smoothed_pca_dx
                pca_dy_np = smoothed_pca_dy
                pca_mean_x_np = smoothed_pca_mean_x
                pca_mean_y_np = smoothed_pca_mean_y
                
                print(f"Applied {arrow_smoothing_frames}-frame smoothing to arrow visualization.")
            else:
                print("Warning: PCA DataFrame provided but required columns are missing. No arrow will be drawn.")
        else:
            print("No PCA data provided for arrow visualization.")

        # --- Prepare median coordinates for overlay ---
        median_x_np = None
        median_y_np = None
        if median_coords is not None and not median_coords.empty:
            if ('analysis', 'median_x') in median_coords.columns and \
               ('analysis', 'median_y') in median_coords.columns:
                median_x_np = median_coords[('analysis', 'median_x')].to_numpy()
                median_y_np = median_coords[('analysis', 'median_y')].to_numpy()
                print("Median coordinates provided for overlay.")
            else:
                print("Warning: Median coordinates DataFrame provided but required columns are missing.")
        else:
            print("No median coordinates provided for overlay.")

        # --- Prepare all time series data ---
        speed_np = speed_data.fillna(0).to_numpy()
        body_movement_np = body_movement_derivative.fillna(0).to_numpy()
        angular_velocity_np = angular_velocity_data.fillna(0).to_numpy()
        angle_data_np = smoothed_angle_data.fillna(0).to_numpy()
        
        # Convert angles to 0-360 format for polar plot
        angles_0_360 = ((angle_data_np + 180) % 360)
        angles_radians = np.radians(angles_0_360)

        # --- Prepare sleep periods from each method ---
        speed_periods = []
        if df_speed_sleep_bouts is not None and not df_speed_sleep_bouts.empty:
            for _, bout in df_speed_sleep_bouts.iterrows():
                speed_periods.append((bout['start_time_s'], bout['end_time_s']))
            print(f"Added {len(speed_periods)} speed-based sleep periods.")
        
        posture_periods = []
        if df_posture_sleep_bouts is not None and not df_posture_sleep_bouts.empty:
            for _, bout in df_posture_sleep_bouts.iterrows():
                posture_periods.append((bout['start_time_s'], bout['end_time_s']))
            print(f"Added {len(posture_periods)} posture-based sleep periods.")
        
        angular_periods = []
        if df_angular_sleep_bouts is not None and not df_angular_sleep_bouts.empty:
            for _, bout in df_angular_sleep_bouts.iterrows():
                angular_periods.append((bout['start_time_s'], bout['end_time_s']))
            print(f"Added {len(angular_periods)} angular velocity-based sleep periods.")

        # --- Function to draw median point and arrow on each frame ---
        def draw_median_and_arrow_on_frame(get_frame, t):
            frame_orig = get_frame(t)
            frame = frame_orig.copy()
            current_frame_idx = int(t * actual_frame_rate)
            
            # Draw median point if data is available
            if median_x_np is not None and median_y_np is not None and \
            current_frame_idx < len(median_x_np) and current_frame_idx < len(median_y_np):
                
                mx = median_x_np[current_frame_idx]
                my = median_y_np[current_frame_idx]

                if not np.isnan(mx) and not np.isnan(my):
                    center_coordinates = (int(mx), int(my))
                    try:
                        cv2.circle(frame, center_coordinates, median_point_radius, median_point_color, -1)
                    except Exception as e_cv2:
                        print(f"Error drawing circle at frame {current_frame_idx}: {e_cv2}")
            
            # Draw arrow if PCA data is available
            if have_pca_data and current_frame_idx < len(pca_dx_np):
                dx = pca_dx_np[current_frame_idx]
                dy = pca_dy_np[current_frame_idx]
                mean_x = pca_mean_x_np[current_frame_idx]
                mean_y = pca_mean_y_np[current_frame_idx]
                
                if not np.isnan(dx) and not np.isnan(dy) and not np.isnan(mean_x) and not np.isnan(mean_y):
                    try:
                        # Get coordinates for anatomical reference from bodypart_coordinate_sets
                        # We need to access the original raw dataframes
                        coords_dfs_x = bodypart_coordinate_sets["axial_plus_midpoints"]["x"]
                        coords_dfs_y = bodypart_coordinate_sets["axial_plus_midpoints"]["y"]
                        
                        # Use a combination of anatomical points to determine front direction
                        front_reference_parts = ['neck', 'ears_midpoint', 'eartips_midpoint']
                        back_reference_part = 'mid_backend'
                        
                        front_x = None
                        front_y = None
                        back_x = None
                        back_y = None
                        
                        # Try to get back reference point
                        if back_reference_part in coords_dfs_x.columns and back_reference_part in coords_dfs_y.columns:
                            back_x = coords_dfs_x.loc[current_frame_idx, back_reference_part]
                            back_y = coords_dfs_y.loc[current_frame_idx, back_reference_part]
                        
                        # Try each front reference part until we find a valid one
                        for front_part in front_reference_parts:
                            if front_part in coords_dfs_x.columns and front_part in coords_dfs_y.columns:
                                temp_x = coords_dfs_x.loc[current_frame_idx, front_part]
                                temp_y = coords_dfs_y.loc[current_frame_idx, front_part]
                                
                                if not pd.isna(temp_x) and not pd.isna(temp_y):
                                    front_x = temp_x
                                    front_y = temp_y
                                    break
                        
                        # If we have valid front and back references, use them to orient the arrow
                        if (front_x is not None and front_y is not None and 
                            back_x is not None and back_y is not None and
                            not pd.isna(front_x) and not pd.isna(front_y) and
                            not pd.isna(back_x) and not pd.isna(back_y)):
                            
                            # Calculate anatomical direction vector
                            anat_dx, anat_dy = front_x - back_x, front_y - back_y
                            
                            # Check if PCA direction needs to be flipped to match anatomical direction
                            if (dx * anat_dx + dy * anat_dy) < 0:
                                dx, dy = -dx, -dy
                        
                        # Calculate the line endpoints
                        x1 = int(mean_x - arrow_length * dx)
                        y1 = int(mean_y - arrow_length * dy)
                        x2 = int(mean_x + arrow_length * dx)
                        y2 = int(mean_y + arrow_length * dy)
                        
                        # Draw the body axis line
                        cv2.line(frame, (x1, y1), (x2, y2), arrow_color, 2)
                        
                        # Draw arrowhead at the "front" end of the line
                        arrow_angle = np.arctan2(y2 - y1, x2 - x1)
                        arrow_x1 = int(x2 - arrow_size * np.cos(arrow_angle - np.pi/6))
                        arrow_y1 = int(y2 - arrow_size * np.sin(arrow_angle - np.pi/6))
                        arrow_x2 = int(x2 - arrow_size * np.cos(arrow_angle + np.pi/6))
                        arrow_y2 = int(y2 - arrow_size * np.sin(arrow_angle + np.pi/6))
                        
                        # Draw the arrowhead lines
                        cv2.line(frame, (x2, y2), (arrow_x1, arrow_y1), arrow_color, 2)
                        cv2.line(frame, (x2, y2), (arrow_x2, arrow_y2), arrow_color, 2)
                    except Exception as e_cv2:
                        print(f"Error drawing arrow at frame {current_frame_idx}: {e_cv2}")
                        
            return frame

        # Apply the drawing function to the video clip
        video_clip_processed = video_clip_orig.fl(draw_median_and_arrow_on_frame)

        # --- Set up plot scaling ---
        valid_speeds = speed_np[np.isfinite(speed_np)]
        max_speed = np.max(valid_speeds) * 1.1 if len(valid_speeds) > 0 else 100.0
        
        valid_body_movement = body_movement_np[np.isfinite(body_movement_np)]
        max_body_movement = np.max(valid_body_movement) * 1.1 if len(valid_body_movement) > 0 else 100.0
        
        valid_angular = angular_velocity_np[np.isfinite(angular_velocity_np)]
        max_angular = np.max(valid_angular) * 1.1 if len(valid_angular) > 0 else 100.0

        # Calculate window parameters
        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        # --- Create main figure with 3 subplots ---
        fig, (ax_speed, ax_body, ax_angular) = plt.subplots(3, 1, figsize=(w / 80, plot_height_pixels * 3 / 80), sharex=True, dpi=80)
        
        # Speed plot (no grid)
        line_speed, = ax_speed.plot([], [], color='gray')
        ax_speed.set_ylabel("Speed (px/s)")
        ax_speed.set_ylim(0, max_speed)
        ax_speed.axhline(speed_threshold, color='r', linestyle='--', lw=0.5, alpha=0.5, label=f'Threshold: {speed_threshold}')
        ax_speed.legend(loc='upper right', fontsize='x-small')
        
        # Body movement derivative plot (no grid)
        line_body, = ax_body.plot([], [], color='purple')
        ax_body.set_ylabel("Body Movement (px/s)")
        ax_body.set_ylim(0, max_body_movement)
        ax_body.axhline(posture_threshold, color='r', linestyle='--', lw=0.5, alpha=0.5, label=f'Threshold: {posture_threshold}')
        ax_body.legend(loc='upper right', fontsize='x-small')
        
        # Angular velocity plot (no grid)
        line_angular, = ax_angular.plot([], [], color='blue')
        ax_angular.set_xlabel("Time (s)")
        ax_angular.set_ylabel("Angular Velocity (°/s)")
        ax_angular.set_ylim(0, max_angular)
        ax_angular.axhline(angular_threshold, color='r', linestyle='--', lw=0.5, alpha=0.5, label=f'Threshold: {angular_threshold}')
        ax_angular.legend(loc='upper right', fontsize='x-small')
        
        # Add current time marker to all plots
        vline_speed = ax_speed.axvline(0, color='lime', linestyle='--', lw=1)
        vline_body = ax_body.axvline(0, color='lime', linestyle='--', lw=1)
        vline_angular = ax_angular.axvline(0, color='lime', linestyle='--', lw=1)
        
        fig.tight_layout(pad=0.5)
        
        # --- Create polar figure (with static setup) ---
        polar_fig = plt.figure(figsize=(h/4 / 80, h/4 / 80), dpi=80)
        polar_ax = polar_fig.add_subplot(111, projection='polar')
        
        # Set up polar plot with static elements
        polar_ax.set_theta_zero_location('N')
        polar_ax.set_theta_direction(-1)
        polar_ax.set_rorigin(0)
        polar_ax.set_rmin(0)
        polar_ax.set_rmax(1.0)
        polar_ax.grid(True, alpha=0.3)
        polar_ax.set_rticks([])
        polar_ax.set_xticks(np.radians([0, 90, 180, 270]))
        polar_ax.set_xticklabels(['0°', '90°', '±180°', '-90°'], fontsize='xx-small')
        
        # Set figure background to transparent
        polar_fig.patch.set_alpha(0)
        
        # --- PRE-CALCULATE SLEEP PERIODS OVERLAP SPANS ---
        # Calculate all time points in the video duration at 0.1s intervals
        time_points = np.arange(0, vid_duration, 0.1)
        sleep_spans_data = []
        
        print("Pre-calculating sleep period overlaps...")
        
        # Helper function to determine overlapping sleep periods
        def is_in_all_sleep_periods(t):
            in_speed = any(start <= t < end for start, end in speed_periods) if speed_periods else False
            in_posture = any(start <= t < end for start, end in posture_periods) if posture_periods else False
            in_angular = any(start <= t < end for start, end in angular_periods) if angular_periods else False
            
            methods_with_data = [bool(speed_periods), bool(posture_periods), bool(angular_periods)]
            methods_with_sleep = [in_speed, in_posture, in_angular]
            
            if all(methods_with_data) and all(methods_with_sleep):
                return True
                
            if sum(methods_with_data) == 2:
                active_methods = [m for i, m in enumerate(methods_with_sleep) if methods_with_data[i]]
                if all(active_methods):
                    return True
                    
            if sum(methods_with_data) == 1:
                active_methods = [m for i, m in enumerate(methods_with_sleep) if methods_with_data[i]]
                if all(active_methods):
                    return True
                    
            return False
        
        # Find continuous sleep spans
        if time_points.size > 0:
            in_span = False
            span_start = None
            
            for t in time_points:
                is_sleep = is_in_all_sleep_periods(t)
                
                if is_sleep and not in_span:
                    span_start = t
                    in_span = True
                elif not is_sleep and in_span:
                    sleep_spans_data.append((span_start, t))
                    in_span = False
            
            # Don't forget the last span if it extends to the end
            if in_span:
                sleep_spans_data.append((span_start, vid_duration))
                
            print(f"Pre-calculated {len(sleep_spans_data)} sleep overlap periods.")
        else:
            print("No time points available for pre-calculation.")

        # Function to create plot frames with pre-calculated spans
        def make_plot_frame(t):
            current_frame = int(t * actual_frame_rate)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = min(len(speed_np), current_frame + 1)
            
            # Extract data segments for the plotting window
            time_axis_segment = np.arange(start_frame_idx, end_frame_idx) * time_per_frame
            speed_segment = speed_np[start_frame_idx:end_frame_idx]
            body_segment = body_movement_np[start_frame_idx:end_frame_idx]
            angular_segment = angular_velocity_np[start_frame_idx:end_frame_idx]
            
            # Set up plot window limits
            plot_window_end_time = t + (time_per_frame * 5)
            plot_window_start_time = max(0.0, plot_window_end_time - plot_width_seconds)
            
            if t < plot_width_seconds - (time_per_frame * 5):
                plot_window_start_time = 0.0
                plot_window_end_time = plot_width_seconds
            
            # Set X limits for all plots
            ax_speed.set_xlim(plot_window_start_time, plot_window_end_time)
            
            # Update line data
            line_speed.set_data(time_axis_segment, speed_segment)
            line_body.set_data(time_axis_segment, body_segment)
            line_angular.set_data(time_axis_segment, angular_segment)
            
            # Update current time markers
            vline_speed.set_xdata([t, t])
            vline_body.set_xdata([t, t])
            vline_angular.set_xdata([t, t])
            
            # Clear any previous spans
            for ax in [ax_speed, ax_body, ax_angular]:
                for artist in list(ax.patches):  # Specifically target patches (spans)
                    artist.remove()
            
            # Draw sleep spans that are visible in the current window
            for span_start, span_end in sleep_spans_data:
                if span_end >= plot_window_start_time and span_start <= plot_window_end_time:
                    visible_start = max(span_start, plot_window_start_time)
                    visible_end = min(span_end, plot_window_end_time)
                    
                    ax_speed.axvspan(visible_start, visible_end, color='palegreen', alpha=0.3, zorder=0)
                    ax_body.axvspan(visible_start, visible_end, color='palegreen', alpha=0.3, zorder=0)
                    ax_angular.axvspan(visible_start, visible_end, color='palegreen', alpha=0.3, zorder=0)
            
            # Draw the main plot
            fig.canvas.draw()
            img_buf = fig.canvas.buffer_rgba()
            plot_img = np.frombuffer(img_buf, dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plot_img = plot_img[:, :, :3]  # Drop alpha channel
            
            # Clear the polar plot for the new frame
            polar_ax.clear()
            
            # Re-apply static polar plot settings (minimal)
            polar_ax.set_theta_zero_location('N')
            polar_ax.set_theta_direction(-1)
            polar_ax.set_rticks([])
            polar_ax.set_xticks(np.radians([0, 90, 180, 270]))
            polar_ax.set_xticklabels(['0°', '90°', '±180°', '-90°'], fontsize='xx-small')
            
            # Calculate progression for the polar plot (radius based on time)
            progress = min(t / vid_duration, 1.0)
            
            # Draw angle trace up to current time
            if current_frame > 0:
                # Select only up to current frame
                angles_subset = angles_radians[:current_frame+1]
                
                # Create radii that increase with time
                radii = np.linspace(0, progress, len(angles_subset))
                
                # Plot the line
                polar_ax.plot(angles_subset, radii, color='blue', linewidth=1.5)
                
                # Mark the current angle point
                if current_frame < len(angles_radians):
                    current_angle = angles_radians[current_frame]
                    polar_ax.plot([current_angle], [progress], marker='o', color='red', markersize=5)
            
            polar_fig.canvas.draw()
            
            polar_buf = polar_fig.canvas.buffer_rgba()
            polar_img = np.frombuffer(polar_buf, dtype=np.uint8)
            polar_img = polar_img.reshape(polar_fig.canvas.get_width_height()[::-1] + (4,))
            polar_img = polar_img[:, :, :3]  # Drop alpha channel
            
            return plot_img, polar_img

        # Create the plot clip
        def make_combined_frame(t):
            plot_img, polar_img = make_plot_frame(t)
            return plot_img
        
        plot_clip = mpy.VideoClip(make_combined_frame, duration=vid_duration)
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels * 3))
        
        # Create polar plot clip
        def make_polar_frame(t):
            _, polar_img = make_plot_frame(t)
            return polar_img
        
        polar_size = int(h/4)
        polar_clip = mpy.VideoClip(make_polar_frame, duration=vid_duration)
        polar_clip = polar_clip.resize(newsize=(polar_size, polar_size))
        
        # Combine video and polar plot
        def combine_video_and_polar(t):
            video_frame = video_clip_processed.get_frame(t)
            polar_frame = polar_clip.get_frame(t)
            
            margin = 10
            y_offset = margin
            x_offset = w - polar_size - margin
            
            y_end = y_offset + polar_size
            x_end = x_offset + polar_size
            
            if y_end <= h and x_end <= w:
                alpha = 0.7
                video_frame[y_offset:y_end, x_offset:x_end] = (
                    (1 - alpha) * video_frame[y_offset:y_end, x_offset:x_end] + 
                    alpha * polar_frame
                ).astype(np.uint8)
            
            return video_frame
        
        video_with_polar = mpy.VideoClip(combine_video_and_polar, duration=vid_duration)
        
        # Combine video and plot
        final_clip = mpy.clips_array([[video_with_polar], [plot_clip]])

        # Write the video file
        print(f"Writing final video to {output_video_path}...")
        try:
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='h264_nvenc', audio=False,
                threads=4, preset='fast', logger='bar'
            )
            print("Video encoding with h264_nvenc successful.")
        except Exception as e_nvenc:
            print(f"Warning: h264_nvenc encoding failed ({e_nvenc}). Falling back to libx264.")
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='libx264', audio=False,
                threads=4, preset='ultrafast', logger='bar'
            )

        # Clean up resources
        if fig is not None:
            plt.close(fig)
        if polar_fig is not None:
            plt.close(polar_fig)
        video_clip_orig.close()
        print("Optimized comprehensive sleep analysis video complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and fig is not None:
            plt.close(fig)
        if 'polar_fig' in locals() and polar_fig is not None:
            plt.close(polar_fig)
        if 'video_clip_orig' in locals() and hasattr(video_clip_orig, 'close'):
            video_clip_orig.close()
    finally:
        matplotlib.use(original_backend)
        print(f"Restored matplotlib backend to: {original_backend}")


