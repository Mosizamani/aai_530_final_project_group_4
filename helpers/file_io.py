import os
import pickle
import pandas as pd

from typing import Any

def load_subject_pickle(pkl_path: str) -> dict[str, Any]:
    """Load a subject's pickle file containing raw sensor data.
    
    Args:
        pkl_path: Path to the pickle file to load.
        
    Returns:
        Dictionary containing subject data with 'signal' and 'label' keys.
        The 'signal' key contains physiological sensor readings (ECG, EDA, 
        Respiration, Temperature, EMG, Accelerometer). The 'label' key 
        contains ground truth stress annotations.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

def validate_subject_files(subject_path: str, subject: str) -> tuple[str, str] | None:
    """
    Check if both pickle and questionnaire files exist for a subject.
    
    Args:
        subject_path: Path to the subject's directory
        subject: Subject identifier (e.g., 'S2')
        
    Returns:
        Tuple of (pkl_path, quest_path) if both exist, None otherwise
    """
    pkl_path = os.path.join(subject_path, f'{subject}.pkl')
    quest_path = os.path.join(subject_path, f'{subject}_quest.csv')
    
    if not os.path.exists(pkl_path) or not os.path.exists(quest_path):
        print(f"Skipping {subject}: missing pkl or quest file")
        return None
    
    return pkl_path, quest_path

def load_subject_questionnaire(quest_path: str) -> pd.DataFrame:
    """Load a subject's questionnaire CSV file."""
    return pd.read_csv(quest_path)

def create_output_directory(subject_path: str) -> str:
    """
    Create cleaned_data directory for a subject.
    
    Args:
        subject_path: Path to the subject's directory
        
    Returns:
        Path to the cleaned_data directory
    """
    cleaned_data_dir = os.path.join(subject_path, 'cleaned_data')
    os.makedirs(cleaned_data_dir, exist_ok=True)
    return cleaned_data_dir

def save_event_timings(df_event_timings: pd.DataFrame, output_dir: str) -> None:
    """Save event timings DataFrame to CSV."""
    output_path = os.path.join(output_dir, 'df_event_timings.csv')
    df_event_timings.to_csv(output_path, index=False)

def save_questionnaire_responses(df_questionnaire_responses: pd.DataFrame, output_dir: str) -> None:
    """Save questionnaire responses DataFrame to CSV."""
    output_path = os.path.join(output_dir, 'df_questionnaire_responses.csv')
    df_questionnaire_responses.to_csv(output_path, index=False)

def save_sensor_dataframes(sensor_dataframes: dict[str, pd.DataFrame], sensor_type: str, output_dir: str) -> None:
    """
    Save all sensor DataFrames for a specific sensor type (chest or wrist).
    
    Args:
        sensor_dataframes: Dictionary of sensor_name -> DataFrame
        sensor_type: 'chest' or 'wrist'
        output_dir: Directory to save the CSV files
    """
    for sensor_name, df in sensor_dataframes.items():
        filename = f'{sensor_type}_{sensor_name.lower()}.csv'
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)

def save_all_cleaned_data(output_dir: str, df_event_timings: pd.DataFrame, 
                          df_questionnaire_responses: pd.DataFrame, 
                          chest_dataframes: dict[str, pd.DataFrame], 
                          wrist_dataframes: dict[str, pd.DataFrame]) -> None:
    """
    Save all cleaned data (event timings, questionnaires, and sensor data) to CSV files.
    
    Args:
        output_dir: Directory to save all files
        df_event_timings: Event timings DataFrame
        df_questionnaire_responses: Questionnaire responses DataFrame
        chest_dataframes: Dictionary of chest sensor DataFrames
        wrist_dataframes: Dictionary of wrist sensor DataFrames
    """
    save_event_timings(df_event_timings, output_dir)
    save_questionnaire_responses(df_questionnaire_responses, output_dir)
    save_sensor_dataframes(chest_dataframes, 'chest', output_dir)
    save_sensor_dataframes(wrist_dataframes, 'wrist', output_dir)
