import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import glob

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


class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed Geolife trajectories.
    """
    def __init__(self, data_dir, sequence_length):
        """
        Args:
            data_dir (str): Directory containing the preprocessed .npy files.
            sequence_length (int): The length of the sequences to be returned.
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        search_pattern = os.path.join(self.data_dir, "**", "*.npy")
        self.file_paths = glob.glob(search_pattern, recursive=True)
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        for file_path in self.file_paths:
            trajectory = np.load(file_path)
            if len(trajectory) >= self.sequence_length:
                for i in range(len(trajectory) - self.sequence_length + 1):
                    seq = trajectory[i:i + self.sequence_length]
                    sequences.append(seq)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.from_numpy(sequence).float()

def get_dataloaders(data_dir, sequence_length, batch_size, val_split=0.1, test_split=0.1, seed=42):
    """
    Creates and returns training, validation, and test DataLoaders.

    Args:
        data_dir (str): Directory with preprocessed data.
        sequence_length (int): Length of input sequences.
        batch_size (int): The batch size for the DataLoaders.
        val_split (float): The proportion of the dataset to use for validation.
        test_split (float): The proportion of the dataset to use for testing.
        seed (int): Random seed for reproducible splits.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    set_seed(seed)
    dataset = TrajectoryDataset(data_dir, sequence_length)

    # Split dataset
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size

    # Ensure the splits add up to the total dataset size to avoid errors
    if train_size + val_size + test_size != len(dataset):
        train_size = len(dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the dataloader
    processed_data_dir = os.path.join('../data', 'processed')
    seq_len = 50
    batch_sz = 64
    RANDOM_SEED = 42

    set_seed(RANDOM_SEED)

    # Ensure you have preprocessed data before running this
    if not os.path.exists(processed_data_dir) or not os.listdir(processed_data_dir):
        print("Processed data not found. Please run preprocess.py first.")
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            processed_data_dir, seq_len, batch_sz, seed=RANDOM_SEED
        )

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")

        # Check a sample batch
        for batch in train_loader:
            print(f"Sample batch shape: {batch.shape}")
            break

