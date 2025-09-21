import numpy as np
import torch
import torch.nn as nn
import os

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # For CUDA set seed and deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def calculate_threshold(model, data_loader, device):
    """
    Finds a suitable anomaly threshold from the training data's reconstruction errors.
    """
    model.eval()
    losses = []
    criterion = nn.MSELoss(reduction='none') # We want to calculate loss per sample
    print("Calculating anomaly threshold...")
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructions = model(batch)
            # Calculate loss for each sequence in the batch
            loss = criterion(reconstructions, batch).mean(dim=[1, 2])
            losses.extend(loss.cpu().numpy())
    
    losses = np.array(losses)
    mean = np.mean(losses)
    std = np.std(losses)
    print(f"mean: {mean}, std: {std}")
    # A common approach is to set the threshold based on the distribution of losses
    threshold = np.mean(losses) + 3 * np.std(losses)
    return threshold

def check_above_threshold(model, data_loader, threshold, device):
    """
    Detects anomalies in the test set based on the calculated threshold.
    """
    model.eval()
    anomalies_found = 0
    total_samples = 0
    criterion = nn.MSELoss(reduction='none')
    print("Detecting anomalies in the test set...")
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructions = model(batch)
            loss = criterion(reconstructions, batch).mean(dim=[1, 2])
            
            # Compare loss to the threshold to find anomalies
            anomalies = loss > threshold
            anomalies_found += anomalies.sum().item()
            total_samples += batch.size(0)
    
    print(f"\n--- Anomaly Detection Results ---")
    print(f"Found {anomalies_found} anomalies out of {total_samples} samples.")
    print(f"Anomaly rate: {100 * anomalies_found / total_samples:.2f}%")
    
