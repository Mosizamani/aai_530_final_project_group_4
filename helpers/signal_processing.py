import pandas as pd
import numpy as np
from typing import Any

def process_accelerometer_data(data_array: np.ndarray) -> pd.DataFrame:
    """
    Process 3-axis accelerometer data into a DataFrame with x, y, z columns.
    
    Accelerometer data from WESAD is stored as an Nx3 array where each row
    contains [x, y, z] acceleration values. This converts it to a DataFrame
    with labeled columns for easier analysis.
    
    Args:
        data_array: Numpy array of shape (n_samples, 3) containing x, y, z acceleration
        
    Returns:
        DataFrame with columns 'ACC_x', 'ACC_y', 'ACC_z'
    """
    return pd.DataFrame(data_array, columns=['ACC_x', 'ACC_y', 'ACC_z'])

def process_single_channel_sensor(data_array: np.ndarray, sensor_name: str) -> pd.DataFrame:
    """
    Process single-channel sensor data into a DataFrame.
    
    Many WESAD sensors (ECG, EDA, BVP, temperature, respiration) record a single
    value per sample. This function flattens the array (if needed) and creates
    a DataFrame with the sensor name as the column.
    
    Args:
        data_array: Numpy array containing sensor readings
        sensor_name: Name of the sensor (e.g., 'ECG', 'EDA', 'BVP')
        
    Returns:
        DataFrame with one column named after the sensor
    """
    return pd.DataFrame(data_array.flatten(), columns=[sensor_name])

def process_sensor_signals(sensor_data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Process all sensor signals from a chest or wrist sensor dictionary.
    
    WESAD data contains multiple sensors on both chest and wrist devices:
    - Chest: ECG, EDA, EMG, Temp, Resp, ACC (3-axis)
    - Wrist: BVP, EDA, TEMP, ACC (3-axis)
    
    This function iterates through all sensors and converts them to DataFrames,
    handling 3-axis accelerometer data differently from single-channel sensors.
    After processing, all DataFrame indices are reset to 0-based integers for
    consistency in downstream analysis.
    
    Args:
        sensor_data: Dictionary mapping sensor names to numpy arrays
        
    Returns:
        Dictionary mapping sensor names to processed DataFrames with reset indices
    """
    dataframes: dict[str, pd.DataFrame] = {}
    
    for sensor_name, data_array in sensor_data.items():
        if sensor_name == 'ACC':
            dataframes[sensor_name] = process_accelerometer_data(data_array)
        else:
            dataframes[sensor_name] = process_single_channel_sensor(data_array, sensor_name)

    for sensor_name in dataframes:
        dataframes[sensor_name] = dataframes[sensor_name].reset_index(drop=True)
    
    return dataframes

def process_physiological_signals(sensor_data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Process physiological sensor signals (chest or wrist) into DataFrames.
    
    Main entry point for processing WESAD sensor data. Converts raw sensor
    data arrays into pandas DataFrames with appropriate column names and
    clean 0-based indices. Handles both 3-axis accelerometer data and
    single-channel sensors automatically.
    
    Args:
        sensor_data: Dictionary mapping sensor names to numpy arrays
                    (e.g., {'ACC': array, 'ECG': array, 'EDA': array})
        
    Returns:
        Dictionary mapping sensor names to processed DataFrames with reset indices
    """
    dataframes = process_sensor_signals(sensor_data)

    return dataframes


def calculate_recording_duration(wrist_dataframes: dict[str, pd.DataFrame], 
                                sampling_rate: float = 64.0) -> float | None:
    """
    Calculate recording duration based on BVP (Blood Volume Pulse) samples.
    
    BVP is used as the reference because it's consistently recorded at 64 Hz
    across all subjects in WESAD. Typical recording duration is ~90 minutes
    per subject (including baseline, stress tasks, and recovery periods).
    
    Args:
        wrist_dataframes: Dictionary containing wrist sensor DataFrames
        sampling_rate: BVP sampling rate in Hz (default: 64.0)
        
    Returns:
        Duration in minutes, or None if BVP data not found
    """
    if 'BVP' in wrist_dataframes:
        duration_min = len(wrist_dataframes['BVP']) / sampling_rate / 60.0
        return duration_min
    return None

def log_recording_duration(subject: str, wrist_dataframes: dict[str, pd.DataFrame], 
                          sampling_rate: float = 64.0) -> None:
    """
    Calculate and print recording duration for a subject.
    
    Provides a sanity check during data processing to verify that subjects
    have complete recordings (~90 minutes expected). Helps identify any
    truncated or corrupted data files.
    
    Args:
        subject: Subject identifier (e.g., 'S2', 'S3')
        wrist_dataframes: Dictionary containing wrist sensor DataFrames
        sampling_rate: BVP sampling rate in Hz (default: 64.0)
    """
    duration_min = calculate_recording_duration(wrist_dataframes, sampling_rate)
    if duration_min is not None:
        print(f"  {subject}: {len(wrist_dataframes['BVP'])} BVP samples = {duration_min:.1f} min")
