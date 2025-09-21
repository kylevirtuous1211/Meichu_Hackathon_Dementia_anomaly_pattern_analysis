import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Dementia_Meichu.model.dataloader import get_dataloaders
from Dementia_Meichu.deploy.anomaly_util import calculate_threshold

class TrajectoryAutoencoder(nn.Module):
    """
    An LSTM-based autoencoder for trajectory data.
    """
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super(TrajectoryAutoencoder, self).__init__()
        # --- Encoder ---
        self.encoder = nn.LSTM(
            input_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        # --- Decoder ---
        # The decoder's input and hidden dimensions should match the encoder's hidden dimension
        # to properly process the context vector.
        self.decoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            n_layers, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        # --- Reconstruction Layer ---
        # A fully connected layer to map the decoder's output back to the original input dimension.
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Encode the input sequence
        # hidden shape: (n_layers, batch_size, hidden_dim)
        _, (hidden, cell) = self.encoder(x)

        # We repeat it 'seq_len' times to create the decoder input sequence.
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        # decoder_input shape: (batch_size, seq_len, hidden_dim)
        decoder_outputs, _ = self.decoder(decoder_input, (hidden, cell))
        # decoder_outputs shape: (batch_size, seq_len, hidden_dim)
        reconstructions = self.fc(decoder_outputs)
        # reconstructions shape: (batch_size, seq_len, input_dim)
        
        return reconstructions

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_dir, log_interval=100):
    """
    Trains the autoencoder model and returns the loss history.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_val_loss = float('inf')
    best_model_path = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            # add gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

            if (i + 1) % log_interval == 0:
                print(f'Train Epoch: {epoch + 1} [{i * len(batch)}/{len(train_loader.dataset)} '
                      f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save a checkpoint for the current epoch
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Saved checkpoint: {epoch_checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Remove previous best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            new_best_model_path = os.path.join(checkpoint_dir, f'best_checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), new_best_model_path)
            best_model_path = new_best_model_path
            print(f"New best model saved to {best_model_path}")

    return best_model_path, train_loss_history, val_loss_history

def plot_loss_history(train_losses, val_losses, save_path):
    """
    Plots the training and validation loss curves and saves the plot to a file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"\nLoss plot saved to: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    # --- Configuration ---
    PROCESSED_DATA_DIR = os.path.join('../data', 'processed_scaled_deltas')
    CHECKPOINT_DIR = '../checkpoints/lstm_deltas'
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 64
    INPUT_DIM = 3  # Latitude, Longitude, and Time Delta
    HIDDEN_DIM = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    LOG_INTERVAL = 500
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    RESUME_CHECKPOINT_PATH = '/home/cvlab123/Kyle_Having_Fun/Dementia_Meichu/checkpoints/lstm_deltas/checkpoint_epoch_10.pth'
    
    print(f"Using device: {DEVICE}")
    print(f"Saving checkpoint to {CHECKPOINT_DIR}")
    print(f"Training data dir use: {PROCESSED_DATA_DIR}")

    # --- Get DataLoaders ---
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print(f"Processed data not found in '{PROCESSED_DATA_DIR}'.")
        print("Please run preprocess.py first.")
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            PROCESSED_DATA_DIR, SEQUENCE_LENGTH, BATCH_SIZE
        )

        # --- Initialize Model and Optimizer ---
        model = TrajectoryAutoencoder(INPUT_DIM, HIDDEN_DIM)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_epoch = 0

        # --- Load from Checkpoint if specified ---
        if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
            checkpoint = torch.load(RESUME_CHECKPOINT_PATH)
            print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
            model.load_state_dict(checkpoint)
            # Try to infer start epoch from filename
            try:
                filename = os.path.basename(RESUME_CHECKPOINT_PATH)
                epoch_num_str = filename.split('_')[-1].split('.')[0]
                start_epoch = int(epoch_num_str)
                print(f"Inferred start epoch from filename: {start_epoch}. Resuming training from epoch {start_epoch + 1}.")
            except (IndexError, ValueError):
                print("Could not infer epoch number from filename. Starting from epoch 1.")
                start_epoch = 0
            print(f"Loaded model and optimizer states. Starting from epoch {start_epoch + 1}.")
        else:
            print("Starting training from scratch.")
            
        best_model_path, train_losses, val_losses = train_model(
            model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE, CHECKPOINT_DIR, LOG_INTERVAL
        )
        print(f"\nModel training has finished, best model is at {best_model_path}")
        
        # --- Plot and Save Loss History ---
        if train_losses and val_losses:
            plot_path = os.path.join(CHECKPOINT_DIR, 'loss_history_plot.png')
            plot_loss_history(train_losses, val_losses, plot_path)

        # --- Calculate and Save Anomaly Threshold ---
        print("\n--- Calculating and Saving Anomaly Threshold ---")
        if best_model_path:
            # Load the best model to calculate the threshold
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
            model.to(DEVICE)
            
            # Calculate threshold on the original training data
            print("Calculating anomaly threshold on training data...")
            threshold = calculate_threshold(model, train_loader, DEVICE)
            
            threshold_path = os.path.join(CHECKPOINT_DIR, 'threshold_lstm.txt')
            with open(threshold_path, 'w') as f:
                f.write(str(threshold))
            
            print(f"Anomaly threshold {threshold:.6f} saved to {threshold_path}")
        else:
            print("No best model was saved. Skipping threshold calculation.")
