import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import joblib
import time
from torch.utils.data import TensorDataset, DataLoader

# Import the model class from your existing script
from Dementia_Meichu.model.lstm import TrajectoryAutoencoder
from Dementia_Meichu.deploy.find_anomaly_threshold import find_anomaly_threshold

def reconstruct_path_from_deltas(scaled_deltas, scaler):
    """
    Takes scaled deltas and reconstructs the absolute lat/lon path.
    """
    # 1. Inverse transform the scaling
    unscaled_deltas = scaler.inverse_transform(scaled_deltas)
    
    # 2. Cumulatively sum the deltas to get the path
    # We assume the path starts at (0, 0) for visualization purposes
    # as the absolute start point is not stored in the delta file.
    path = np.cumsum(unscaled_deltas[:, :2], axis=0) # Only sum lat and lon
    
    return path

def create_sequences_from_data(scaled_deltas, sequence_length):
    """
    Creates overlapping sequences from the scaled delta data.
    """
    sequences = []
    for i in range(len(scaled_deltas) - sequence_length + 1):
        sequences.append(scaled_deltas[i:i + sequence_length])
        
    if not sequences:
        return None
        
    return torch.tensor(np.array(sequences), dtype=torch.float32)

def get_point_anomaly_scores(model, sequences_tensor, device, sequence_length):
    """
    Runs inference and maps sequence errors back to individual points.
    """
    model.eval()
    dataset = TensorDataset(sequences_tensor)
    # Use a larger batch size for faster inference
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            reconstructions = model(batch)
            errors = torch.mean((reconstructions - batch) ** 2, dim=[1, 2])
            all_errors.extend(errors.cpu().numpy())

    # Map sequence errors back to individual points.
    # A point's anomaly score is the max error of any sequence it's a part of.
    num_points = len(sequences_tensor) + sequence_length - 1
    point_scores = np.zeros(num_points)
    
    for i, seq_error in enumerate(all_errors):
        for j in range(sequence_length):
            point_index = i + j
            point_scores[point_index] = max(point_scores[point_index], seq_error)
            
    return point_scores

def plot_trajectory_with_anomalies(reconstructed_path, anomaly_scores, threshold, output_path):
    """
    Plots the reconstructed GPS path with anomaly scores as a heatmap.
    """
    longitudes = reconstructed_path[:, 1]
    latitudes = reconstructed_path[:, 0]

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use the square root of scores for better color contrast
    color_scores = np.sqrt(anomaly_scores)
    
    # Create the scatter plot with a color map
    sc = ax.scatter(longitudes, latitudes, c=color_scores, cmap='plasma', s=15, vmin=0, vmax=np.sqrt(threshold * 2))
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Anomaly Score (sqrt of Reconstruction Error)')
    
    ax.plot(longitudes[0], latitudes[0], 'go', markersize=12, label='Start')
    ax.plot(longitudes[-1], latitudes[-1], 'ro', markersize=12, label='End')
    
    plot_title = os.path.basename(output_path).replace('_heatmap.png', '')
    ax.set_title(f'GPS Trajectory with Anomaly Heatmap for: {plot_title}')
    ax.set_xlabel('Relative Longitude')
    ax.set_ylabel('Relative Latitude')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == '__main__':
    # --- Configuration ---
    # Directory with the preprocessed and SCALED .npy files
    INPUT_DATA_DIR = '../data/processed_testing_anomalous' 
    OUTPUT_PLOT_DIR = '../plots/anomaly_heatmaps'
    
    CHECKPOINT_DIR = '../checkpoints/lstm_deltas'

    # --- Paths to saved artifacts ---
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_23.pth') 
    # THRESHOLD_PATH = os.path.join(CHECKPOINT_DIR, 'threshold_percentile_99.txt')
    SCALER_PATH = os.path.join(INPUT_DATA_DIR, 'scaler.gz') # The scaler is in the same dir as the data

    # --- Model Hyperparameters (MUST match training) ---
    INPUT_DIM = 3
    HIDDEN_DIM = 64
    SEQUENCE_LENGTH = 50
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Script Start ---
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # --- 1. Load Model, Scaler, and Threshold ---
    try:
        model = TrajectoryAutoencoder(INPUT_DIM, HIDDEN_DIM)
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        
        scaler = joblib.load(SCALER_PATH)
        
        # with open(THRESHOLD_PATH, 'r') as f:
        #     anomaly_threshold = float(f.read().strip())
        anomaly_threshold = find_anomaly_threshold(INPUT_DATA_DIR, percentile=50)
        
        print("Model, scaler, and threshold loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. Please check your paths.")
        exit()

    # --- 2. Find and Process .npy Files ---
    # Search recursively for all .npy files
    npy_files = glob.glob(os.path.join(INPUT_DATA_DIR, '**', '*.npy'), recursive=True)
    print(f"Found {len(npy_files)} trajectory files to visualize.")

    for npy_file in tqdm(npy_files, desc="Generating Heatmap Plots"):
        # --- 3. Load Data and Create Sequences ---
        scaled_deltas = np.load(npy_file)
        
        if len(scaled_deltas) < SEQUENCE_LENGTH:
            continue
        
        sequences_tensor = create_sequences_from_data(scaled_deltas, SEQUENCE_LENGTH)
        if sequences_tensor is None:
            continue
            
        # --- 4. Get Anomaly Scores ---
        anomaly_scores = get_point_anomaly_scores(model, sequences_tensor, DEVICE, SEQUENCE_LENGTH)
        
        # --- 5. Reconstruct Path for Plotting ---
        reconstructed_path = reconstruct_path_from_deltas(scaled_deltas, scaler)
        
        # --- 6. Generate and Save Plot ---
        # Create a more descriptive filename
        user_id = os.path.basename(os.path.dirname(npy_file))
        base_name = os.path.basename(npy_file).replace('.npy', '')
        output_filename = f"{user_id}_{base_name}_heatmap.png"
        output_path = os.path.join(OUTPUT_PLOT_DIR, output_filename)
        
        plot_trajectory_with_anomalies(reconstructed_path, anomaly_scores, anomaly_threshold, output_path)

    print("\nVisualization complete.")
    print(f"All heatmap plots saved in '{os.path.abspath(OUTPUT_PLOT_DIR)}'")

