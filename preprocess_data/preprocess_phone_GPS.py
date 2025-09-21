import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

def load_and_calculate_deltas(log_file_path):
    """
    Reads a single raw .log file and calculates the deltas.
    """
    raw_data_points = []
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.strip('[]').split(',')
                    if len(parts) == 3:
                        try:
                            point = [float(p.strip()) for p in parts]
                            raw_data_points.append(point)
                        except ValueError:
                            pass # Silently skip malformed lines
    except Exception as e:
        print(f"Could not read file {log_file_path}: {e}")
        return None

    if len(raw_data_points) < 2:
        return None

    # Process raw data into deltas
    deltas = [[0.0, 0.0, 0.0]]
    for i in range(1, len(raw_data_points)):
        delta_lat = raw_data_points[i][0] - raw_data_points[i-1][0]
        delta_lon = raw_data_points[i][1] - raw_data_points[i-1][1]
        delta_time = raw_data_points[i][2] - raw_data_points[i-1][2]
        deltas.append([delta_lat, delta_lon, delta_time])
    
    return np.array(deltas, dtype=np.float32)

def preprocess_all_logs(input_dir, output_dir):
    """
    Processes all .log files in a directory, fits a scaler,
    saves the scaled data, and saves the scaler.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_files = glob.glob(os.path.join(input_dir, '*.log'))
    if not log_files:
        print(f"No .log files found in '{input_dir}'.")
        return

    print(f"Found {len(log_files)} log files. First pass: calculating all deltas...")
    
    # --- First Pass: Collect all deltas from all files ---
    all_deltas = []
    trajectory_data_map = {} # Store individual trajectory deltas

    for log_file in tqdm(log_files, desc="Calculating Deltas"):
        deltas = load_and_calculate_deltas(log_file)
        if deltas is not None:
            all_deltas.append(deltas)
            trajectory_data_map[log_file] = deltas

    if not all_deltas:
        print("No valid data found in any log files.")
        return

    # Combine all deltas into a single large array for fitting the scaler
    combined_deltas = np.concatenate(all_deltas, axis=0)

    # --- Fit the Scaler ---
    print("\nFitting StandardScaler on all collected data...")
    scaler = StandardScaler()
    scaler.fit(combined_deltas)
    
    # --- Save the Scaler ---
    scaler_path = os.path.join(output_dir, 'scaler.gz')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # --- Second Pass: Transform and Save Individual Files ---
    print("\nSecond pass: Scaling and saving individual trajectory files...")
    for log_file, deltas in tqdm(trajectory_data_map.items(), desc="Saving Scaled Files"):
        scaled_deltas = scaler.transform(deltas)
        
        base_name = os.path.basename(log_file).replace('.log', '.npy')
        output_path = os.path.join(output_dir, base_name)
        
        np.save(output_path, scaled_deltas)
        
    print(f"\nPreprocessing complete. All scaled .npy files are in '{output_dir}'")


if __name__ == '__main__':
    # --- Configuration ---
    # 1. Directory containing your raw .log files
    INPUT_LOGS_DIR = '../data/test_raw_GPS/correct'  # <--- CHANGE THIS
    
    # 2. Directory where the scaled .npy files and scaler.gz will be saved
    OUTPUT_PROCESSED_DIR = '../data/processed_testing_correct' # <--- CHANGE THIS

    # --- Run Processing ---
    preprocess_all_logs(INPUT_LOGS_DIR, OUTPUT_PROCESSED_DIR)
