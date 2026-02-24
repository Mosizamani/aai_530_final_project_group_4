# VitalGuard: Real-Time Wearable Stress & Affect Detection System
    
**VitalGuard** is an end-to-end IoT pipeline designed to detect physiological stress using wearable sensor data. The project leverages the **WESAD (Wearable Stress and Affect Detection)** multimodal dataset to benchmark three distinct machine learning architectures for real-time monitoring and detection.

## Authors
* **Michael Domingo**
* **Eric Hernandez**
* **Mostafa Zamaniturk**

## Project Overview
The core of this project is to evaluate the robustness of different modeling approaches for stress detection. We implemented and compared three distinct machine learning pipelines:
1. **1D-Convolutional Neural Network (1D-CNN)**: Functions as a pattern recognition engine on raw BVP and EDA waveforms.
2. **Long Short-Term Memory (LSTM)**: Optimized for sequential data and temporal pattern recognition.
3. **XGBoost (Gradient Boosting)**: Utilizes statistical feature engineering for fast and efficient classification.

## Dataset & Sensors
The project utilizes the **WESAD multimodal dataset**, extracting data from laboratory-grade and consumer-grade wearable sensors:
* **Chest Sensors (RespiBAN, 700 Hz)**: ACC, ECG, EMG, EDA, TEMP, RESP.
* **Wrist Sensors (Empatica E4)**: ACC (32 Hz), BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz).
* **Metadata**: Includes experimental event timings (#ORDER, #START, #END) and questionnaire responses (PANAS, STAI, DIM, SSSQ).

## Technical Implementation
### Preprocessing Pipeline
* **De-duplication**: Removed massive duplicate rows (e.g., ~4.2M from Chest TEMP) to ensure data integrity.
* **Frequency Alignment**: All sensors were synchronized to a uniform **64 Hz target** using scipy.signal.resample and linear interpolation.
* **Normalization**: Applied subject-specific **Z-score normalization** to account for individual physiological baselines.

## Repository Structure
* `data/compressed_dataset`: dedicated directory for the WESAD compressed file and aligned wrist data for 1d-cnn model 
* `1d-cnn/`: Data cleaning, exploration, and model results and analysis.
* `lstm/`: Data cleaning, exploration, and model results and analysis.
* `XGBoost:` Data cleaning, exploration, and model results and analysis
* `aai_530_final_project.ipynb:` Main notebook containing all 3 models and results

## Instructions for use

Import the data set:
1. Download the compressed dataset [here](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
2. Save zip file in the dedicated directory `data/compressed_dataset`
3. Extract aligned_wrist_data_64Hz_v1.csv.zip located in data and save in 1d-cnn/ directory

Install dependencies:
1. Run: 
```bash
pip install -r requirements.txt
```
Run notebook aai_530_final_project.ipynb
