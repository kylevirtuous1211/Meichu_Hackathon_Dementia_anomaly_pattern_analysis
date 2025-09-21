import os
import torch
import joblib
import numpy as np
from tqdm import tqdm
import glob
from torch.utils.data import DataLoader

# We assume the dataloader file contains a class named 'TrajectoryDataset'
# If the class has a different name, you may need to adjust the import.
from Dementia_Meichu.model.dataloader import TrajectoryDataset
from Dementia_Meichu.model.lstm import TrajectoryAutoencoder
from Dementia_Meichu.deploy.find_anomaly_threshold import find_anomaly_threshold

def detect_anomalies(model, test_loader, threshold, device, consecutive_threshold=10):
    """
    Infers on the test data and classifies sustained anomalous events.
    This version only prints the start and end of an event for cleaner output.
    """
    model.eval()
    anomaly_event_count = 0
    total_count = 0
    consecutive_anomalies = 0
    event_start_index = -1

    print(f"\n--- Starting Anomaly Detection (Event Threshold: {consecutive_threshold} consecutive anomalies) ---")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inferencing")):
            batch = batch.to(device)
            reconstructions = model(batch)
            errors = torch.mean((reconstructions - batch) ** 2, dim=[1, 2])
            
            for j, error in enumerate(errors):
                current_index = total_count + j
                is_anomaly = error.item() > threshold
                
                if is_anomaly:
                    # If it's an anomaly, just increment the counter.
                    consecutive_anomalies += 1
                    20250920-185130
                    # Check if we have just reached the threshold to start a new event.
                    if consecutive_anomalies == consecutive_threshold:
                        event_start_index = current_index - consecutive_threshold + 1
                        anomaly_event_count += 1
                        print(f"\n--- ANOMALOUS EVENT #{anomaly_event_count} STARTED at Trajectory Index: {event_start_index} ---")

                else: # This block only runs if the signal is NORMAL
                    # If an event was in progress and now we see a normal signal, the event has ended.
                    print(f"[info] signal is normal: ")
                    if consecutive_anomalies >= consecutive_threshold:
                        event_end_index = current_index - 1
                        duration = event_end_index - event_start_index + 1
                        print(f"--- Event #{anomaly_event_count} ENDED at Trajectory Index: {event_end_index} (Duration: {duration} steps) ---\n")
                    
                    # Reset the counter because the streak of anomalies is broken.
                    consecutive_anomalies = 0
                    event_start_index = -1

            total_count += len(batch)

    # Check if the file ends during an anomaly event
    if consecutive_anomalies >= consecutive_threshold:
        event_end_index = total_count - 1
        duration = event_end_index - event_start_index + 1
        print(f"--- Event #{anomaly_event_count} ENDED at Trajectory Index: {event_end_index} (Duration: {duration} steps) ---\n")

    print("\n--- Detection Complete ---")
    print(f"Total Trajectories Tested: {total_count}")
    print(f"Total Anomalous Events Found: {anomaly_event_count}")

if __name__ == '__main__':
    # --- Configuration (MUST match your training setup) ---
    PROCESSED_DATA_DIR = os.path.join('../data', 'processed_testing_anomalous/')
    # PROCESSED_DATA_DIR = os.path.join('../data', 'processed_testing_correct')
    # TODO: set the model checkpoint dir
    CHECKPOINT_DIR = '../checkpoints/lstm_deltas'
    
    # --- Paths to saved artifacts ---
    # TODO: set the model checkpoint file
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_23.pth') 
    # TODO: set the percentile .txt filename
    THRESHOLD_PATH = os.path.join(PROCESSED_DATA_DIR, 'threshold_percentile_e23_99_9.txt')
    SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'scaler.gz')

    # --- Model & Dataloader Hyperparameters ---
    INPUT_DIM = 3
    HIDDEN_DIM = 64
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 128 # Can be different from training, but 128 is efficient
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {BEST_MODEL_PATH}")
    print(f"Loading threshold from: {THRESHOLD_PATH}")
    print(f"Loading scaler from: {SCALER_PATH}")
    
    # --- 1. calculate the Threshold ---
    anomaly_threshold = find_anomaly_threshold(PROCESSED_DATA_DIR, percentile=90)
    print(f"threshold is: {anomaly_threshold}")
    # try:
    #     with open(THRESHOLD_PATH, 'r') as f:
    #         anomaly_threshold = float(f.read().strip())
    # except FileNotFoundError:
    #     print(f"Error: Threshold file not found at {THRESHOLD_PATH}")
    #     exit()

    # --- 2. Load the Model ---
    try:
        model = TrajectoryAutoencoder(INPUT_DIM, HIDDEN_DIM)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {BEST_MODEL_PATH}")
        exit()

    # --- 3. Load the Scaler ---
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Scaler object not found at {SCALER_PATH}")
        print("You must run the `preprocess_data.py` script first.")
        exit()
        
    # --- 4. Get the DataLoader with ALL data ---
    # We now bypass the train/val/test split to use all available data for testing.
    print("\n--- Creating DataLoader with ALL available data for testing ---")
    try:
        # Create a single dataset with all the files found
        full_dataset = TrajectoryDataset(PROCESSED_DATA_DIR, SEQUENCE_LENGTH)
        
        # Create a DataLoader for the full dataset. Shuffle is False for consistent inference order.
        test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        print(f"Created a single test dataset of length {len(full_dataset)}.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error creating DataLoader: {e}")
        print(f"Please ensure the directory '{PROCESSED_DATA_DIR}' contains preprocessed .npy files.")
        exit()

    # --- 5. Run Anomaly Detection ---
    detect_anomalies(model, test_loader, anomaly_threshold, DEVICE)

