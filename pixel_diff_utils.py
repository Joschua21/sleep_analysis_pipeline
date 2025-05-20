import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_video_file(input_dir, filename=None):
    """
    Find a video file in the given directory.
    
    Args:
        input_dir: Directory containing video file(s)
        filename: Specific filename to look for (optional)
        
    Returns:
        Path to the video file
    """
    if filename and os.path.exists(os.path.join(input_dir, filename)):
        return os.path.join(input_dir, filename)
    
    # Search for video files with preferred extensions
    video_extensions = ['.mj2', '.mp4', '.avi']
    video_files = [f for f in os.listdir(input_dir) 
                  if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        raise FileNotFoundError(f"No video files found in {input_dir}")
    
    # Prefer mj2 if available
    mj2_files = [f for f in video_files if f.lower().endswith('.mj2')]
    if mj2_files:
        video_file = os.path.join(input_dir, mj2_files[0])
        print(f"Using MJ2 file: {mj2_files[0]}")
    else:
        # Next preference is mp4
        mp4_files = [f for f in video_files if f.lower().endswith('.mp4')]
        if mp4_files:
            video_file = os.path.join(input_dir, mp4_files[0])
            print(f"Using MP4 file: {mp4_files[0]}")
        else:
            # Last resort is any other video file
            video_file = os.path.join(input_dir, video_files[0])
            print(f"Using video file: {video_files[0]}")
    
    return video_file

def calculate_pixel_differences(input_dir, filename=None, smooth_window=15, black_buffer=5, 
                               output_dir=None, black_threshold=5.0):
    """
    Calculate pixel-by-pixel differences between consecutive frames in a video.
    Includes black frame detection and handling by replacing with NaN values.
    
    Args:
        input_dir: Directory containing the video file
        filename: Specific video filename to analyze (optional)
        smooth_window: Window size for smoothing the differences (# of frames)
        black_buffer: Number of frames before and after black frames to also mark as NaN
        output_dir: Directory to save output files (defaults to input_dir/pixel_difference)
        black_threshold: Brightness threshold for black frame detection
        
    Returns:
        Dictionary with analysis results
    """
    # Find the video file
    input_video_path = find_video_file(input_dir, filename)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "pixel_difference")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input video: {input_video_path}")
    print(f"Output directory: {output_dir}")
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {frame_count} frames, {fps} fps, {width}x{height} pixels")
    print(f"Video duration: {frame_count / fps:.2f} seconds")
    
    # Create arrays to store frame differences and frame information
    differences = np.zeros(frame_count - 1)
    time_seconds = np.arange(frame_count - 1) / fps
    
    # Arrays to store frame brightness for black frame detection
    mean_brightness = np.zeros(frame_count)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Could not read the first frame")
    
    # Store information about the first frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mean_brightness[0] = np.mean(prev_gray)
    
    # Process each subsequent frame
    print("Calculating frame differences...")
    
    with tqdm(total=frame_count-1) as pbar:
        frame_idx = 0
        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean brightness of current frame
            current_brightness = np.mean(curr_gray)
            mean_brightness[frame_idx+1] = current_brightness
            
            # Calculate absolute difference between frames
            frame_diff = cv2.absdiff(curr_gray, prev_gray)
            
            # Sum all pixel differences to get total motion
            differences[frame_idx] = np.sum(frame_diff)
            
            # Update previous frame
            prev_gray = curr_gray
            
            # Update counter
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"Processed {frame_idx} frame differences")
    
    # Detect black frames using the specified threshold
    is_black_frame = mean_brightness < black_threshold
    
    # Count black frames
    black_frame_count = np.sum(is_black_frame)
    print(f"Detected {black_frame_count} potential black frames")
    
    # Find indices of black frames for debugging
    black_frame_indices = np.where(is_black_frame)[0]
    if len(black_frame_indices) > 0:
        print(f"Black frames found at frame indices: {black_frame_indices}")

    # Create a mask for transitions involving black frames
    mask_to_nan = np.zeros_like(differences, dtype=bool)
    
    # Mark the affected frame differences (transitions)
    for i in range(len(differences)):
        # Check if either the current frame or next frame is black
        if i < len(is_black_frame)-1 and (is_black_frame[i] or is_black_frame[i+1]):
            mask_to_nan[i] = True
    
    # Expand the mask to include buffer frames
    if black_buffer > 0:
        expanded_mask = mask_to_nan.copy()
        for i in np.where(mask_to_nan)[0]:  # Only process marked frames
            # Add buffer before
            start = max(0, i - black_buffer)
            # Add buffer after
            end = min(len(mask_to_nan) - 1, i + black_buffer)
            expanded_mask[start:end+1] = True
        
        mask_to_nan = expanded_mask
    
    # Count affected transitions
    nan_transitions = np.sum(mask_to_nan)
    print(f"Removing {nan_transitions} frame transitions due to black frames (including buffer)")
    
    # Replace black frame transitions and buffer with NaN
    differences_filtered = differences.copy()
    differences_filtered[mask_to_nan] = np.nan
    
    # Apply smoothing only to valid segments (between NaN regions)
    valid_mask = ~np.isnan(differences_filtered)
    smoothed_diffs = np.full_like(differences_filtered, np.nan)  # Initialize with all NaNs
    
    # Identify continuous segments of valid data
    valid_regions = []
    in_valid_region = False
    start_idx = 0
    
    for i in range(len(valid_mask)):
        if valid_mask[i] and not in_valid_region:
            # Start of a valid region
            start_idx = i
            in_valid_region = True
        elif not valid_mask[i] and in_valid_region:
            # End of a valid region
            valid_regions.append((start_idx, i))
            in_valid_region = False
    
    # Handle case where video ends during a valid region
    if in_valid_region:
        valid_regions.append((start_idx, len(valid_mask)))
    
    print(f"Found {len(valid_regions)} continuous valid segments for smoothing")
    
    # Apply smoothing to each valid segment independently
    for start, end in valid_regions:
        segment_len = end - start
        
        # Only smooth if the segment is long enough
        if segment_len >= smooth_window:
            segment_data = differences_filtered[start:end]
            kernel = np.ones(smooth_window) / smooth_window
            
            # Apply convolution for smoothing
            smoothed_segment = np.convolve(segment_data, kernel, mode='same')
            
            # Store the smoothed data
            smoothed_diffs[start:end] = smoothed_segment
    
    # Calculate the derivative (only where we have valid data)
    derivative = np.zeros(len(smoothed_diffs) - 1)
    derivative[:] = np.nan
    
    for i in range(len(derivative)):
        if not np.isnan(smoothed_diffs[i]) and not np.isnan(smoothed_diffs[i+1]):
            derivative[i] = smoothed_diffs[i+1] - smoothed_diffs[i]
    
    derivative_time = time_seconds[:-1]  # Time points for derivative
    
    # Absolute derivative
    abs_derivative = np.abs(derivative)
    
    # Create base filename from input video
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # Save the raw data
    data_df = pd.DataFrame({
        'time_sec': time_seconds,
        'frame_num': np.arange(1, frame_count),
        'frame_brightness': mean_brightness[1:],  # Brightness of each frame
        'is_black_frame': is_black_frame[1:],     # Whether frame is black
        'raw_difference': differences,            # Raw pixel difference
        'filtered_difference': differences_filtered,  # With NaNs at black frames
        'smoothed_difference': smoothed_diffs,    # Smoothed data
        'is_black_or_buffer': mask_to_nan         # Mask of all affected frames
    })
    
    # Add derivative to dataframe (will have one fewer row)
    derivative_df = pd.DataFrame({
        'time_sec': derivative_time,
        'derivative': derivative,
        'abs_derivative': abs_derivative
    })
    
    # Save data to CSV files
    data_path = os.path.join(output_dir, f"{base_name}_pixel_differences.csv")
    data_df.to_csv(data_path, index=False)
    
    print(f"Saved difference data to: {data_path}")
    
    # Calculate summary statistics ignoring NaN values
    valid_smoothed = smoothed_diffs[~np.isnan(smoothed_diffs)]
    valid_derivative = abs_derivative[~np.isnan(abs_derivative)]
    
    smoothed_mean = np.mean(valid_smoothed) if len(valid_smoothed) > 0 else np.nan
    smoothed_max = np.max(valid_smoothed) if len(valid_smoothed) > 0 else np.nan
    derivative_mean = np.mean(valid_derivative) if len(valid_derivative) > 0 else np.nan
    derivative_max = np.max(valid_derivative) if len(valid_derivative) > 0 else np.nan
    
    summary_stats = {
        'mean_motion': smoothed_mean,
        'max_motion': smoothed_max,
        'mean_rate_of_change': derivative_mean,
        'max_rate_of_change': derivative_max,
        'black_frame_count': black_frame_count,
        'black_transitions_count': nan_transitions
    }
    
    return {
        'differences': differences,
        'filtered_differences': differences_filtered,
        'smoothed': smoothed_diffs,
        'time_seconds': time_seconds,
        'derivative': derivative,
        'abs_derivative': abs_derivative,
        'derivative_time': derivative_time,
        'fps': fps,
        'frame_count': frame_count,
        'output_dir': output_dir,
        'data_path': data_path,
        'black_frames': black_frame_indices,
        'black_transitions': np.where(mask_to_nan)[0],
        'summary_stats': summary_stats,
        'base_name': base_name
    }

def plot_pixel_differences(results=None, csv_path=None, y_min=None, y_max=None, 
                          show_black_frames=False, plot_type='smoothed', sleep_threshold=675000,
                          save_fig=True, show_fig=True, output_dir=None, input_dir=None,
                          min_sleep_duration=15):
    """
    Plot pixel difference data from either analysis results or a CSV file.
    
    Args:
        results: Results dictionary from calculate_pixel_differences (optional)
        csv_path: Path to CSV file with pixel differences (optional)
        y_min: Minimum y-axis value (optional)
        y_max: Maximum y-axis value (optional)
        show_black_frames: Whether to highlight black frames
        plot_type: Type of data to plot ('raw', 'filtered', or 'smoothed')
        sleep_threshold: Threshold for pixel difference to indicate sleep
        save_fig: Whether to save the figure
        show_fig: Whether to display the figure
        output_dir: Directory to save figure (defaults to results['output_dir'])
        input_dir: Alternative way to specify the input directory for auto-finding CSV
        min_sleep_duration: Minimum duration in seconds to consider as sleep
        save_sleep_periods: Whether to save sleep periods to CSV
        
    Returns:
        Path to saved figure or sleep periods data if save_sleep_periods is True
    """
    # If neither results nor csv_path provided, try to find the most recent CSV
    if results is None and csv_path is None:
        if input_dir:
            # Look for pixel_difference directory
            pixel_diff_dir = os.path.join(input_dir, "pixel_difference")
            if not os.path.exists(pixel_diff_dir):
                # If not found, check if input_dir itself is already the pixel_difference directory
                if os.path.basename(input_dir) == "pixel_difference":
                    pixel_diff_dir = input_dir
                else:
                    raise ValueError(f"Could not find pixel_difference directory in {input_dir}")
            
            # Find CSV files in the pixel_difference directory
            csv_files = [f for f in os.listdir(pixel_diff_dir) if f.endswith('_pixel_differences.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No pixel difference CSV files found in {pixel_diff_dir}")
            
            # Use the first CSV file found (could also sort by modification time)
            csv_path = os.path.join(pixel_diff_dir, csv_files[0])
            print(f"Auto-selected CSV file: {csv_path}")
        else:
            # Check if there's a global 'results' variable
            global_results = globals().get('results')
            if global_results and isinstance(global_results, dict) and 'data_path' in global_results:
                csv_path = global_results['data_path']
                print(f"Using CSV from previous results: {csv_path}")
            else:
                raise ValueError("Must provide either results, csv_path, or input_dir")
    
    if results is not None:
        time_seconds = results['time_seconds']
        if plot_type == 'raw':
            differences = results['differences']
            y_label = 'Raw Pixel Difference'
            title_suffix = 'Raw'
        elif plot_type == 'filtered':
            differences = results['filtered_differences']
            y_label = 'Filtered Pixel Difference'
            title_suffix = 'Filtered'
        else:  # Default to smoothed
            differences = results['smoothed']
            y_label = 'Smoothed Pixel Difference'
            title_suffix = 'Smoothed'
            
        black_frames = results.get('black_frames', [])
        fps = results.get('fps', 30)
        output_dir = output_dir or results.get('output_dir', '.')
        base_name = results.get('base_name', 'video')
        
    elif csv_path is not None:
        # Load data from CSV
        data = pd.read_csv(csv_path)
        time_seconds = data['time_sec']
        
        if plot_type == 'raw':
            differences = data['raw_difference']
            y_label = 'Raw Pixel Difference'
            title_suffix = 'Raw'
        elif plot_type == 'filtered':
            differences = data['filtered_difference']
            y_label = 'Filtered Pixel Difference'
            title_suffix = 'Filtered'
        else:  # Default to smoothed
            differences = data['smoothed_difference']
            y_label = 'Smoothed Pixel Difference'
            title_suffix = 'Smoothed'
            
        # Extract black frame info if available
        black_frames = []
        if 'is_black_frame' in data.columns:
            black_frames = data.index[data['is_black_frame']].tolist()
        
        # Get FPS by assuming frame numbers are consecutive
        fps = len(data) / (data['time_sec'].iloc[-1] - data['time_sec'].iloc[0])
        
        output_dir = output_dir or os.path.dirname(csv_path)
        base_name = os.path.basename(csv_path).split('_')[0]
    
    # Detect potential sleep periods (just for visualization, not saving)
    is_sleep = differences < sleep_threshold
    sleep_periods = []
    in_sleep = False
    start_idx = 0
    
    for i in range(len(is_sleep)):
        if is_sleep[i] and not in_sleep:
            # Start of sleep period
            start_idx = i
            in_sleep = True
        elif (not is_sleep[i] or np.isnan(differences[i])) and in_sleep:
            # End of sleep period (or NaN encountered)
            end_idx = i - 1
            duration_sec = time_seconds[end_idx] - time_seconds[start_idx]
            
            # Only count as sleep if duration exceeds minimum
            if duration_sec >= min_sleep_duration:
                sleep_periods.append({
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'start_time_sec': time_seconds[start_idx],
                    'end_time_sec': time_seconds[end_idx],
                    'duration_sec': duration_sec
                })
            in_sleep = False
    
    # Handle case where video ends during sleep
    if in_sleep:
        end_idx = len(is_sleep) - 1
        duration_sec = time_seconds[end_idx] - time_seconds[start_idx]
        if duration_sec >= min_sleep_duration:
            sleep_periods.append({
                'start_frame': start_idx,
                'end_frame': end_idx,
                'start_time_sec': time_seconds[start_idx],
                'end_time_sec': time_seconds[end_idx],
                'duration_sec': duration_sec
            })
    
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # First add sleep period highlighting
    for period in sleep_periods:
        plt.axvspan(period['start_time_sec'], period['end_time_sec'], 
                   color='lightgreen', alpha=0.3)
    
    # Plot pixel differences
    plt.plot(time_seconds, differences, 'b-', linewidth=1)
    
    # Plot sleep threshold line
    plt.axhline(y=sleep_threshold, color='g', linestyle='--', alpha=0.7,
                label=f'Sleep Threshold ({sleep_threshold})')
    
    plt.title(f'Pixel Differences Between Frames ({title_suffix}) - {base_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits if provided
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
        y_range_text = f" (Range: {y_min if y_min is not None else 'min'} to {y_max if y_max is not None else 'max'})"
        plt.title(plt.gca().get_title() + y_range_text)
    
    # Show black frames if requested
    if show_black_frames and len(black_frames) > 0:
        for frame_idx in black_frames:
            if frame_idx < len(time_seconds):
                plt.axvline(time_seconds[frame_idx], color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = None
    if save_fig:
        y_range_str = f"_y_{y_min}_{y_max}" if y_min is not None and y_max is not None else ""
        fig_path = os.path.join(output_dir, f"{base_name}_{plot_type}_pixel_diff{y_range_str}.png")
        plt.savefig(fig_path, dpi=300)
        print(f"Saved figure to: {fig_path}")
    
    if show_fig:
        plt.show()
    else:
        plt.close()
    
    return fig_path

def find_recent_analysis(input_dir=None):
    """
    Find the most recently generated pixel difference analysis in the provided input directory
    or in the last used directory.
    
    Args:
        input_dir: Directory containing the video analysis (optional)
        
    Returns:
        Path to the CSV file containing pixel differences
    """
    # Try to use provided input_dir
    if input_dir is None:
        # Check if we have a global _last_input_dir variable
        global _last_input_dir
        if '_last_input_dir' in globals() and _last_input_dir is not None:
            input_dir = _last_input_dir
            print(f"Using last input directory: {input_dir}")
        else:
            raise ValueError("No input directory provided and no recent analysis found. "
                           "Please provide an input_dir parameter.")
    
    # Look for pixel_difference directory
    pixel_diff_dir = os.path.join(input_dir, "pixel_difference")
    if not os.path.exists(pixel_diff_dir):
        # If not found, check if input_dir itself is already the pixel_difference directory
        if os.path.basename(input_dir) == "pixel_difference":
            pixel_diff_dir = input_dir
        else:
            raise ValueError(f"Could not find pixel_difference directory in {input_dir}")
    
    # Find CSV files in the pixel_difference directory
    csv_files = [f for f in os.listdir(pixel_diff_dir) if f.endswith('_pixel_differences.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No pixel difference CSV files found in {pixel_diff_dir}")
    
    # Sort by modification time (newest first)
    csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(pixel_diff_dir, f)), reverse=True)
    
    # Use the newest CSV file
    csv_path = os.path.join(pixel_diff_dir, csv_files[0])
    print(f"Found analysis file: {csv_path}")
    return csv_path

def load_results(input_dir=None, csv_path=None):
    """
    Load results from a previous analysis without reprocessing the video.
    
    Args:
        input_dir: Directory containing the video analysis (optional)
        csv_path: Direct path to the CSV file (optional)
        
    Returns:
        Dictionary with loaded results
    """
    if csv_path is None:
        csv_path = find_recent_analysis(input_dir)
    
    # Extract base information from the CSV path
    output_dir = os.path.dirname(csv_path)
    base_name = os.path.basename(csv_path).split('_pixel_differences')[0]
    
    # Load the CSV data
    print(f"Loading analysis from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    # Reconstruct a results dictionary similar to what calculate_pixel_differences returns
    results = {
        'differences': data['raw_difference'].values,
        'filtered_differences': data['filtered_difference'].values,
        'smoothed': data['smoothed_difference'].values,
        'time_seconds': data['time_sec'].values,
        'output_dir': output_dir,
        'data_path': csv_path,
        'base_name': base_name,
        'fps': len(data) / (data['time_sec'].iloc[-1] - data['time_sec'].iloc[0])
    }
    
    # Extract black frames if available
    black_frames = []
    if 'is_black_frame' in data.columns:
        black_frames = data.index[data['is_black_frame']].tolist()
    results['black_frames'] = black_frames
    
    # Extract black transitions if available
    black_transitions = []
    if 'is_black_or_buffer' in data.columns:
        black_transitions = data.index[data['is_black_or_buffer']].tolist()
    results['black_transitions'] = black_transitions
    
    print(f"Successfully loaded analysis data with {len(data)} frames")
    return results

def reuse_previous_analysis(y_min=None, y_max=None, sleep_threshold=675000, plot_type='smoothed',
                          min_sleep_duration=15, save_sleep_periods=False, show_black_frames=False,
                          input_dir=None):
    """
    Reuse a previous analysis and create new plots without re-processing the video.
    This function automatically finds the most recent analysis in the specified directory
    or the last used directory.
    
    Args:
        y_min: Minimum y-axis value for plot (optional)
        y_max: Maximum y-axis value for plot (optional)
        sleep_threshold: Threshold for sleep detection
        plot_type: Type of plot to generate ('raw', 'filtered', or 'smoothed')
        min_sleep_duration: Minimum duration in seconds to consider as sleep
        save_sleep_periods: Whether to save sleep periods to CSV
        show_black_frames: Whether to highlight black frames
        input_dir: Directory containing the video analysis (optional)
        
    Returns:
        Path to saved figure or sleep periods data if save_sleep_periods is True
    """
    # Load results from previous analysis
    results = load_results(input_dir)
    
    # Use the loaded results to create a new plot
    return plot_pixel_differences(
        results=results,
        y_min=y_min,
        y_max=y_max,
        sleep_threshold=sleep_threshold,
        plot_type=plot_type,
        min_sleep_duration=min_sleep_duration,
        show_black_frames=show_black_frames
    )


def quick_plot(y_min=None, y_max=None, sleep_threshold=675000, plot_type='smoothed', 
              show_black_frames=False):
    """
    Quickly plot pixel differences using the most recently analyzed video.
    Uses either the global 'results' variable or tries to find the CSV in the last used directory.
    
    Args:
        y_min: Minimum y-axis value (optional)
        y_max: Maximum y-axis value (optional)
        sleep_threshold: Threshold for pixel difference to indicate sleep
        plot_type: Type of data to plot ('raw', 'filtered', or 'smoothed')
        show_black_frames: Whether to highlight black frames
        
    Returns:
        Path to saved figure
    """
    # Check if we have global results from a previous analysis
    global_results = globals().get('results')
    if global_results and isinstance(global_results, dict):
        return plot_pixel_differences(
            results=global_results,
            y_min=y_min,
            y_max=y_max, 
            sleep_threshold=sleep_threshold,
            plot_type=plot_type,
            show_black_frames=show_black_frames
        )
    else:
        # No global results, check if we have a last_input_dir stored
        last_input_dir = globals().get('_last_input_dir')
        if last_input_dir and os.path.exists(last_input_dir):
            return plot_pixel_differences(
                input_dir=last_input_dir,
                y_min=y_min,
                y_max=y_max,
                sleep_threshold=sleep_threshold,
                plot_type=plot_type,
                show_black_frames=show_black_frames
            )
        else:
            raise ValueError("No recent analysis results found. Run analyze_and_plot_video first.")
        
def interactive_sleep_selection(results=None, input_dir=None, sleep_threshold=675000, 
                               min_sleep_duration=15, y_min=None, y_max=None, 
                               plot_type='smoothed', save_file='sleep_times.csv',
                               save_final_plot=True):
    """
    Creates a plot with numbered sleep periods and gets user selection via input.
    
    Args:
        results: Results dictionary from analysis (optional)
        input_dir: Directory containing the video analysis (optional)
        sleep_threshold: Threshold for pixel difference to indicate sleep
        min_sleep_duration: Minimum duration in seconds to consider as sleep
        y_min, y_max: Y-axis limits for the plot
        plot_type: Type of data to plot ('raw', 'filtered', or 'smoothed')
        save_file: Name of the CSV file to save selected periods
        save_final_plot: Whether to save a final plot showing only selected periods
        
    Returns:
        Path to the saved CSV file with selected sleep periods
    """
    # First load results if not provided
    if results is None:
        results = load_results(input_dir)
    
    # Get the appropriate data based on plot type
    time_seconds = results['time_seconds']
    if plot_type == 'raw':
        differences = results['differences']
        y_label = 'Raw Pixel Difference'
        title_suffix = 'Raw'
    elif plot_type == 'filtered':
        differences = results['filtered_differences']
        y_label = 'Filtered Pixel Difference'
        title_suffix = 'Filtered'
    else:  # Default to smoothed
        differences = results['smoothed']
        y_label = 'Smoothed Pixel Difference'
        title_suffix = 'Smoothed'
        
    # Extract base information
    output_dir = results['output_dir']
    base_name = results['base_name']
    
    # Detect potential sleep periods
    is_sleep = differences < sleep_threshold
    sleep_periods = []
    in_sleep = False
    start_idx = 0
    
    for i in range(len(is_sleep)):
        if is_sleep[i] and not in_sleep:
            # Start of sleep period
            start_idx = i
            in_sleep = True
        elif (not is_sleep[i] or np.isnan(differences[i])) and in_sleep:
            # End of sleep period (or NaN encountered)
            end_idx = i - 1
            duration_sec = time_seconds[end_idx] - time_seconds[start_idx]
            
            # Only count as sleep if duration exceeds minimum
            if duration_sec >= min_sleep_duration:
                sleep_periods.append({
                    'start_frame': int(start_idx),
                    'end_frame': int(end_idx),
                    'start_time_s': float(time_seconds[start_idx]),
                    'end_time_s': float(time_seconds[end_idx]),
                    'duration_s': float(duration_sec)
                })
            in_sleep = False
    
    # Handle case where video ends during sleep
    if in_sleep:
        end_idx = len(is_sleep) - 1
        duration_sec = time_seconds[end_idx] - time_seconds[start_idx]
        if duration_sec >= min_sleep_duration:
            sleep_periods.append({
                'start_frame': int(start_idx),
                'end_frame': int(end_idx),
                'start_time_s': float(time_seconds[start_idx]),
                'end_time_s': float(time_seconds[end_idx]),
                'duration_s': float(duration_sec)
            })
    
    if not sleep_periods:
        print("No sleep periods detected to select.")
        return None
    
    import matplotlib.pyplot as plt
    
    # Create a simple figure
    plt.figure(figsize=(8, 5))
    
    # Plot pixel differences
    plt.plot(time_seconds, differences, 'b-', linewidth=1)
    plt.axhline(y=sleep_threshold, color='g', linestyle='--', alpha=0.7,
                label=f'Sleep Threshold ({sleep_threshold})')
    
    # Add sleep period highlighting with numbers
    for i, period in enumerate(sleep_periods):
        # Add span highlighting
        plt.axvspan(period['start_time_s'], period['end_time_s'], 
                   color='lightgreen', alpha=0.3)
        
        # Add a number label in the center of each span
        center_time = (period['start_time_s'] + period['end_time_s']) / 2
        plt.text(center_time, differences[period['start_frame']], str(i+1), 
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f'Sleep Periods - {base_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel(y_label)
    
    # Set y-axis limits if provided
    if y_min is not None or y_max is not None:
        plt.ylim(bottom=y_min, top=y_max)
    
    plt.tight_layout()
    plt.show()
    
    # Print out the sleep periods to the console
    print("\nDetected Sleep Periods:")
    print("-" * 50)
    for i, period in enumerate(sleep_periods):
        print(f"{i+1}. {period['start_time_s']:.1f}s - {period['end_time_s']:.1f}s ({period['duration_s']:.1f}s)")
    print("-" * 50)
    
    # Get input from the user in the console
    selection_str = input("\nEnter sleep periods to include (e.g., 1,3,5-7) or 'all': ")
    
    # Parse the selection string
    selection = []
    if selection_str.strip().lower() == 'all':
        selection = list(range(1, len(sleep_periods) + 1))
    else:
        # Handle individual numbers and ranges (e.g., "1,3,5-7")
        parts = selection_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle range
                try:
                    start, end = map(int, part.split('-'))
                    selection.extend(range(start, end + 1))
                except:
                    print(f"Ignoring invalid range: {part}")
            else:
                # Handle single number
                try:
                    selection.append(int(part))
                except:
                    print(f"Ignoring invalid number: {part}")
    
    # Filter selected periods
    selected_periods = []
    for i, period in enumerate(sleep_periods):
        if i+1 in selection:
            selected_periods.append(period)
    
    # Save to CSV if there are selected periods
    csv_path = None
    if selected_periods:
        selected_df = pd.DataFrame(selected_periods)
        
        # Create the save path
        save_path = os.path.join(output_dir, save_file)
        csv_path = save_path
        
        # Save to CSV
        selected_df.to_csv(save_path, index=False)
        print(f"\nSaved {len(selected_periods)} selected sleep periods to: {save_path}")
        
        # Show summary
        print("\nSelected periods:")
        for i, period in enumerate(selected_periods):
            print(f"{i+1}. {period['start_time_s']:.1f}s - {period['end_time_s']:.1f}s ({period['duration_s']:.1f}s)")
        
        # Create and save a clean final plot with only selected periods highlighted
        if save_final_plot and selected_periods:
            # Create a new figure
            plt.figure(figsize=(8, 5))
            
            # Plot pixel differences
            plt.plot(time_seconds, differences, 'b-', linewidth=1)
            plt.axhline(y=sleep_threshold, color='g', linestyle='--', alpha=0.7,
                        label=f'Sleep Threshold ({sleep_threshold})')
            
            # Add selected sleep period highlighting (no numbers)
            for period in selected_periods:
                plt.axvspan(period['start_time_s'], period['end_time_s'], 
                           color='lightgreen', alpha=0.4)
            
            plt.title(f'Selected Sleep Periods ({len(selected_periods)}) - {base_name}')
            plt.xlabel('Time (seconds)')
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            # Set y-axis limits if provided
            if y_min is not None or y_max is not None:
                plt.ylim(bottom=y_min, top=y_max)
            
            plt.tight_layout()
            
            # Generate the filename
            y_range_str = f"_y_{y_min}_{y_max}" if y_min is not None and y_max is not None else ""
            final_plot_path = os.path.join(output_dir, f"{base_name}_{plot_type}_selected_sleep{y_range_str}.png")
            
            # Save the figure
            plt.savefig(final_plot_path, dpi=300)
            print(f"Saved final plot to: {final_plot_path}")
            plt.close()
            
            # Display the final plot
            plt.figure(figsize=(8, 5))
            img = plt.imread(final_plot_path)
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    else:
        print("\nNo sleep periods selected. Nothing saved.")
    
    # Return the path to the potentially saved file
    return csv_path


def analyze_and_plot_video(input_dir, filename=None, smooth_window=15, 
                          y_min=None, y_max=None, sleep_threshold=675000,
                          plot_type='smoothed', show_fig=True):
    """
    Complete pipeline to analyze video pixel differences and plot the results.
    
    Args:
        input_dir: Directory containing video file
        filename: Specific video file to analyze (optional)
        smooth_window: Window size for smoothing
        y_min: Minimum y-axis value for plot (optional)
        y_max: Maximum y-axis value for plot (optional) 
        sleep_threshold: Threshold for sleep detection
        plot_type: Type of plot to generate ('raw', 'filtered', or 'smoothed')
        show_fig: Whether to display the figure
        
    Returns:
        Dictionary of analysis results
    """
    # Store the input directory for later use by quick_plot
    global _last_input_dir
    _last_input_dir = input_dir
    
    # Calculate pixel differences
    results = calculate_pixel_differences(
        input_dir=input_dir,
        filename=filename,
        smooth_window=smooth_window
    )
    
    # Print summary statistics
    print("\nMotion Analysis Summary:")
    stats = results['summary_stats']
    print(f"  Mean motion: {stats['mean_motion']:.2f}")
    print(f"  Max motion: {stats['max_motion']:.2f}")
    print(f"  Mean rate of change: {stats['mean_rate_of_change']:.2f}")
    print(f"  Max rate of change: {stats['max_rate_of_change']:.2f}")
    print(f"  Black frames detected: {stats['black_frame_count']}")
    print(f"  Transitions affected by black frames: {stats['black_transitions_count']}")
    
    # Plot the results
    plot_pixel_differences(
        results=results,
        y_min=y_min,
        y_max=y_max,
        plot_type=plot_type,
        sleep_threshold=sleep_threshold,
        show_fig=show_fig
    )
    
    return results