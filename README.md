# VitalGuard: Real-Time Wearable Stress & Affect Detection System
    
    **VitalGuard** is an end-to-end IoT pipeline designed to detect physiological stress using wearable sensor data. The project leverages the **WESAD (Wearable Stress and Affect Detection)** multimodal dataset to benchmark three distinct machine learning architectures for real-time monitoring and detection.
    
    ## 👥 Authors
    * **Michael Domingo**
    * **Eric Hernandez**
    * **Mostafa Zamaniturk**
    
    ## 🛠 Project Overview
    The core of this project is to evaluate the robustness of different modeling approaches for stress detection. We implemented and compared three distinct machine learning pipelines:
    1. **1D-Convolutional Neural Network (1D-CNN)**: Functions as a pattern recognition engine on raw BVP and EDA waveforms.
    2. **Long Short-Term Memory (LSTM)**: Optimized for sequential data and temporal pattern recognition.
    3. **XGBoost (Gradient Boosting)**: Utilizes statistical feature engineering for fast and efficient classification.
    
    ## 📊 Dataset & Sensors
    The project utilizes the **WESAD multimodal dataset**, extracting data from laboratory-grade and consumer-grade wearable sensors:
    * **Chest Sensors (RespiBAN, 700 Hz)**: ACC, ECG, EMG, EDA, TEMP, RESP.
    * **Wrist Sensors (Empatica E4)**: ACC (32 Hz), BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz).
    * **Metadata**: Includes experimental event timings (#ORDER, #START, #END) and questionnaire responses (PANAS, STAI, DIM, SSSQ).
    
    ## 🚀 Technical Implementation
    ### Preprocessing Pipeline
    * **De-duplication**: Removed massive duplicate rows (e.g., ~4.2M from Chest TEMP) to ensure data integrity.
    * **Frequency Alignment**: All sensors were synchronized to a uniform **64 Hz target** using scipy.signal.resample and linear interpolation.
    * **Normalization**: Applied subject-specific **Z-score normalization** to account for individual physiological baselines.
    
    ## 📂 Repository Structure
    * `data/`: Processed CSV files tracked via Git LFS.
    * `notebooks/`: Implementation of the 1D-CNN, LSTM, and XGBoost models.
    * `docs/`: Full technical report and system architecture diagrams.
