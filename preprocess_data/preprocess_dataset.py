import os
import glob
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_geolife_data(data_dir, output_dir):
    """
    Reads raw GeoLife .plt files, calculates deltas, fits a StandardScaler
    on ALL the data, then saves the scaled data and the scaler object.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt_files = glob.glob(os.path.join(data_dir, '*', 'Trajectory', '*.plt'))
    print(f"Found {len(plt_files)} trajectory files to process.")

    all_trajectories = []
    output_paths_meta = []

    # --- Pass 1: Read all files and calculate unscaled deltas ---
    for file_path in tqdm(plt_files, desc="Pass 1/2: Reading & Calculating Deltas"):
        points = []
        try:
            with open(file_path, 'r') as f:
                # Skip header lines
                for _ in range(6):
                    next(f)
                
                lines = f.readlines()
                if not lines:
                    continue

                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) == 7:
                        lat, lon = float(parts[0]), float(parts[1])
                        date_str, time_str = parts[5], parts[6]
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        points.append({'lat': lat, 'lon': lon, 'time': timestamp})

            if len(points) < 2:
                continue

            # Calculate deltas for the current trajectory
            deltas = [[0.0, 0.0, 0.0]] # Start with a zero-delta point
            for i in range(1, len(points)):
                delta_lat = points[i]['lat'] - points[i-1]['lat']
                delta_lon = points[i]['lon'] - points[i-1]['lon']
                delta_time = (points[i]['time'] - points[i-1]['time']).total_seconds()
                deltas.append([delta_lat, delta_lon, delta_time])
            
            # Store the unscaled numpy array and its future save path
            all_trajectories.append(np.array(deltas, dtype=np.float32))
            
            user_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            file_name = os.path.basename(file_path).replace('.plt', '.npy')
            user_output_dir = os.path.join(output_dir, user_id)
            output_path = os.path.join(user_output_dir, file_name)
            output_paths_meta.append({'path': output_path, 'dir': user_output_dir})

        except Exception as e:
            print(f"Could not process file {file_path}: {e}")

    if not all_trajectories:
        print("No valid trajectories found to process.")
        return

    # --- Fit the StandardScaler on the entire dataset ---
    print("\nFitting StandardScaler on all collected data points...")
    all_deltas_flat = np.vstack(all_trajectories)
    scaler = StandardScaler()
    scaler.fit(all_deltas_flat)

    # --- Save the scaler object for later use (very important!) ---
    scaler_path = os.path.join(output_dir, 'scaler.gz')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {os.path.abspath(scaler_path)}")

    # --- Pass 2: Transform each trajectory with the fitted scaler and save ---
    for i, trajectory_data in enumerate(tqdm(all_trajectories, desc="Pass 2/2: Transforming & Saving Data")):
        scaled_trajectory = scaler.transform(trajectory_data)
        
        path_info = output_paths_meta[i]
        
        if not os.path.exists(path_info['dir']):
            os.makedirs(path_info['dir'])
        
        np.save(path_info['path'], scaled_trajectory)

if __name__ == '__main__':
    RAW_DATA_PARENT_DIR = '../data/Geolife Trajectories 1.3/Data'
    PROCESSED_DATA_DIR = os.path.join('../data', 'processed_scaled_deltas')
    
    if not os.path.exists(RAW_DATA_PARENT_DIR):
        print(f"Raw data directory not found at '{RAW_DATA_PARENT_DIR}'.")
        print("Please download and extract the GeoLife dataset.")
    else:
        preprocess_geolife_data(RAW_DATA_PARENT_DIR, PROCESSED_DATA_DIR)
        print("\nPreprocessing complete.")
