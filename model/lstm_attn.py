import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from Dementia_Meichu.model.dataloader import get_dataloaders
from Dementia_Meichu.deploy.anomaly_util import calculate_threshold

class Attention(nn.Module):
    """ A simple Bahdanau-style attention mechanism. """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_dim)
        # encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        seq_len = encoder_outputs.size(1)
        
        # Repeat the decoder hidden state for each step of the encoder output
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        
        v_view = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        scores = torch.bmm(v_view, energy).squeeze(1)
        
        # Return the attention weights (softmax)
        return F.softmax(scores, dim=1)

class TrajectoryAutoencoder(nn.Module): # Renamed for consistency in main block
    """ An LSTM Autoencoder with an Attention mechanism. """
    def __init__(self, input_dim, hidden_dim, n_layers=2):
        super(TrajectoryAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2 if n_layers > 1 else 0)
        # Decoder input is concatenation of previous output and context
        self.decoder = nn.LSTM(input_dim + hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2 if n_layers > 1 else 0)
        
        self.attention = Attention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = x.shape
        
        # --- Encoder ---
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # --- Decoder ---
        # The first input to the decoder is a zero tensor
        decoder_input = torch.zeros(batch_size, 1, x.shape[-1]).to(x.device)
        outputs = []

        for t in range(seq_len):
            # Calculate attention weights using the last layer's hidden state
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            
            # Create context vector (weighted sum of encoder outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            
            # Prepare decoder input: concatenate the previous output with the context vector
            rnn_input = torch.cat((decoder_input.squeeze(1), context), dim=1)
            
            # Pass through the decoder LSTM
            decoder_output, (hidden, cell) = self.decoder(rnn_input.unsqueeze(1), (hidden, cell))
            
            # Pass through a final fully connected layer
            output = self.fc_out(decoder_output.squeeze(1))
            outputs.append(output)
            
            # Decide whether to use teacher forcing for the next step
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = x[:, t, :].unsqueeze(1) # Use ground truth
            else:
                decoder_input = output.unsqueeze(1) # Use model's own prediction

        return torch.stack(outputs, dim=1)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, checkpoint_dir, log_interval=100):
    """
    Trains the autoencoder model and returns the loss history.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Add a learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)
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
            # Pass a teacher_forcing_ratio during training
            outputs = model(batch, teacher_forcing_ratio=0.5)
            loss = criterion(outputs, batch)
            loss.backward()
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
                # No teacher forcing during validation
                outputs = model(batch, teacher_forcing_ratio=0.0)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Saved checkpoint: {epoch_checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            new_best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
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
    CHECKPOINT_DIR = '../checkpoints/lstm_attn'
    SEQUENCE_LENGTH = 50
    BATCH_SIZE = 256
    INPUT_DIM = 3  # Latitude, Longitude, and Time Delta
    HIDDEN_DIM = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    LOG_INTERVAL = 200 # Log training loss every 200 batches
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print(f"training data dir use: {PROCESSED_DATA_DIR}")
    print(f"checkpoint at {CHECKPOINT_DIR}")
    print(f"Using device: {DEVICE}")

    # --- Get DataLoaders ---
    if not os.path.exists(PROCESSED_DATA_DIR) or not os.listdir(PROCESSED_DATA_DIR):
        print(f"Processed data not found in '{PROCESSED_DATA_DIR}'.")
        print("Please run preprocess.py first.")
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            PROCESSED_DATA_DIR, SEQUENCE_LENGTH, BATCH_SIZE
        )

        # --- Initialize and Train Model ---
        model = TrajectoryAutoencoder(INPUT_DIM, HIDDEN_DIM)
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
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
            model.to(DEVICE)
            
            print("Calculating anomaly threshold on training data...")
            threshold = calculate_threshold(model, train_loader, DEVICE)
            
            threshold_path = os.path.join(CHECKPOINT_DIR, 'threshold_lstm_attn.txt')
            with open(threshold_path, 'w') as f:
                f.write(str(threshold))
            
            print(f"Anomaly threshold {threshold:.6f} saved to {threshold_path}")
        else:
            print("No best model was saved. Skipping threshold calculation.")

