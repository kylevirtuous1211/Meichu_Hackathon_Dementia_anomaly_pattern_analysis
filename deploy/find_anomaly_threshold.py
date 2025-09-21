import os
import torch
import numpy as np
from tqdm import tqdm
from Dementia_Meichu.model.lstm import TrajectoryAutoencoder
from Dementia_Meichu.model.dataloader import get_dataloaders

def calculate_threshold_percentile(model, data_loader, device, percentile=99.9):
    """
    Calculates the anomaly threshold based on a given percentile of the 
    reconstruction errors from the training data.
    """
    model.eval()
    all_errors = []
    print("\n--- Calculating Reconstruction Errors on Training Data ---")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing Batches"):
            batch = batch.to(device)
            reconstructions = model(batch)
            # Calculate MSE error for each sequence in the batch
            errors = torch.mean((reconstructions - batch) ** 2, dim=[1, 2])
            all_errors.extend(errors.cpu().numpy())
    
    # Calculate the threshold at the specified percentile
    threshold = np.percentile(all_errors, percentile)
    print(f"\nCalculated {percentile}th percentile threshold.")
    return threshold

def find_anomaly_threshold(data_dir, percentile=90):
    # --- Configuration (MUST match your training setup) ---
    ## These define the model architecture and cannot be changed without retraining
    INPUT_DIM = 3  # Latitude, Longitude, and Time Delta
    HIDDEN_DIM = 64
    SEQUENCE_LENGTH = 50
    ## -------------------------------------------------------------------
    BATCH_SIZE = 128
    # # TODO: 改我們'
    PROCESSED_DATA_DIR = data_dir
    # CHECKPOINT_DIR = "../checkpoints/lstm_deltas"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- Load Model ---
    model = TrajectoryAutoencoder(INPUT_DIM, HIDDEN_DIM)
    # TODO: 改這邊
    best_checkpoint_path = '/home/cvlab123/Kyle_Having_Fun/Dementia_Meichu/checkpoints/lstm_deltas/checkpoint_epoch_23.pth'
    print(f"Loading model from: {best_checkpoint_path}")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # --- Calculate Threshold ---
    print("Getting training data to calculate threshold...")
    train_loader, _, _ = get_dataloaders(PROCESSED_DATA_DIR, SEQUENCE_LENGTH, BATCH_SIZE)
    # TODO: find threshold
    threshold = calculate_threshold_percentile(model, train_loader, DEVICE, percentile=percentile)
    
    # --- Save Threshold ---
    # Using a more descriptive filename for the percentile-based threshold
    # threshold_path = os.path.join(PROCESSED_DATA_DIR, 'threshold_percentile_99.txt')
    # with open(threshold_path, 'w') as f:
    #     f.write(str(threshold))
    
    # print(f"Anomaly threshold {threshold:.6f} saved to {threshold_path}")
    return threshold
    
if __name__ == "__main__":
    find_anomaly_threshold()
