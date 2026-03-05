import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def visualize_complex_sequence(X, Y, sample_idx=0, lag=6):
    """
    Visualize a single sequence from a complex sinusoidal dataset.

    Args:
        X (Tensor or np.ndarray): Input lagged sequences (num_samples, lag*2)
        Y (Tensor or np.ndarray): Target values (num_samples, 2)
        sample_idx (int): Index of the sample to visualize
        lag (int): Number of lagged steps
    """
    # Convert tensors to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()

    # Extract the chosen sample
    x_sample = X[sample_idx]
    y_sample = Y[sample_idx]

    # Reconstruct full sequence: lagged inputs + target
    re_sequence = x_sample.real[:lag].tolist() + [y_sample.real]
    im_sequence = x_sample.imag[:lag].tolist() + [y_sample.imag]

    # Compute magnitude and phase
    magnitude = np.sqrt(np.array(re_sequence)**2 + np.array(im_sequence)**2)
    phase = np.arctan2(np.array(im_sequence), np.array(re_sequence))

    time_steps = np.arange(len(re_sequence))

    # 1️⃣ Plot Re and Im
    plt.figure(figsize=(12, 4))
    plt.plot(time_steps, re_sequence, label="Re")
    plt.plot(time_steps, im_sequence, label="Im")
    plt.title("Real and Imaginary Parts vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # 2️⃣ Plot magnitude and phase
    plt.figure(figsize=(12, 4))
    plt.plot(time_steps, magnitude, label="Magnitude")
    plt.plot(time_steps, phase, label="Phase")
    plt.title("Magnitude and Phase vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # 3️⃣ Phase portrait (Re vs Im)
    plt.figure(figsize=(6, 6))
    plt.plot(re_sequence, im_sequence, marker='o')
    plt.title("Phase Portrait (Re vs Im)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.axis('equal')
    plt.show()


def generate_sinusoidal_data(
        num_samples: int = 1400,
        lag: int = 6,
        batch_size: int = 32,
        sigma_r: float = 0.2,
        sigma_phi: float = 0.2,
        seed: int = 42
):
    np.random.seed(seed)
    total_samples = num_samples + lag

    # 1. Generate time array
    t = np.arange(total_samples)

    # 2. Define amplitude R(t) and phase Phi(t)
    R = 1 + 0.5 * np.sin(0.2 * t) + 0.2 * np.sin(0.05 * t)
    Phi = 0.1 + 0.5 * np.sin(0.1 * t) + 0.3 * np.cos(0.07 * t)

    z_clean = R * np.exp(1j * Phi)
    z_noisy = (R + 0.1 * np.random.randn(len(t))) * np.exp(1j * (Phi + 0.1 * np.random.randn(len(t))))

    noise_floor = np.sqrt(np.abs(z_noisy - z_clean) ** 2).mean()
    print(f"Noise floor RMSE: {noise_floor:.4f}")

    # 3. Add Gaussian noise
    R_noisy = R + sigma_r * np.random.randn(total_samples)
    Phi_noisy = Phi + sigma_phi * np.random.randn(total_samples)

    # 4. Construct complex numbers
    z = R_noisy * np.exp(1j * Phi_noisy)  # shape: (num_samples,)

    # 5. Prepare lagged input and target matrices
    X = []
    Y = []
    for i in range(lag, total_samples):
        # lagged inputs
        lagged = z[i - lag:i]
        # convert to two channels: Re and Im
        # lagged_real_imag = np.concatenate([lagged.real, lagged.imag])
        X.append(lagged)

        # target = next step
        # target = np.array([z[i].real, z[i].imag])
        Y.append(z[i])

    X = np.array(X, dtype=np.complex64)
    Y = np.array(Y, dtype=np.complex64)

    # 6. Create PyTorch Dataset and DataLoader
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_dataset,test_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1875, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader, test_loader


if __name__ == "__main__":
    dataset = generate_sinusoidal_data()
    X,y = next(iter(dataset))
    print(X.shape, y.shape)
    visualize_complex_sequence(X, y)