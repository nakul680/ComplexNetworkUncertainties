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
        sigma: float = 0.2,
        seed: int = 42,
        generate_ood: bool = False
):
    np.random.seed(seed)
    total_samples = num_samples + lag

    # 1. Generate time array
    t = np.arange(total_samples)

    # 2. Define amplitude R(t) and phase Phi(t)
    R = 1 + 0.5 * np.sin(0.2 * t) + 0.2 * np.sin(0.05 * t)
    Phi = 0.1 + 0.5 * np.sin(0.1 * t) + 0.3 * np.cos(0.07 * t)

    # 3. Add Gaussian noise
    std = sigma / 2  # scalar, no need for torch here
    R_noisy = R + std * np.random.randn(total_samples)
    Phi_noisy = Phi + std * np.random.randn(total_samples)

    # 4. Construct complex numbers
    z_clean = R * np.exp(1j * Phi)
    z_noisy = R_noisy * np.exp(1j * Phi_noisy)

    noise_floor = np.sqrt(np.abs(z_noisy - z_clean) ** 2).mean()
    print(f"Noise floor RMSE: {noise_floor:.4f}")

    z = z_noisy  # shape: (total_samples,)

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

    if generate_ood:
        ood_samples = 500
        np.random.seed(seed + 1)  # different seed from training data

        # option 1 — pure random complex noise, no temporal structure
        X_rand = (np.random.randn(ood_samples, lag) +
                  1j * np.random.randn(ood_samples, lag)).astype(np.complex64)
        Y_rand = (np.random.randn(ood_samples) +
                  1j * np.random.randn(ood_samples)).astype(np.complex64)

        # option 2 — identity: constant complex value per sample, no dynamics
        constant = np.random.randn(ood_samples) + 1j * np.random.randn(ood_samples)
        X_id = np.tile(constant[:, None], (1, lag)).astype(np.complex64)
        Y_id = constant.astype(np.complex64)

        # combine both OOD types
        X_ood = np.concatenate([X_rand, X_id], axis=0)
        Y_ood = np.concatenate([Y_rand, Y_id], axis=0)

        ood_dataset = TensorDataset(
            torch.from_numpy(X_ood),
            torch.from_numpy(Y_ood)
        )
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, ood_loader

    return train_loader,val_loader, test_loader


def generate_corr_data(
        num_samples: int = 1400,
        lag: int = 6,
        batch_size: int = 32,
        sigma: float = 0.2,
        rho: float = 0.5,   # NEW: correlation control
        seed: int = 42,
        generate_ood: bool = False
):
    np.random.seed(seed)
    total_samples = num_samples + lag

    # 1. Time
    t = np.arange(total_samples)

    # 2. Simple base signals
    x = np.cos(t) + 0.5 * np.cos(0.3 * t)
    y = rho * x + np.sqrt(1 - rho**2) * np.sin(t)

    # 3. Add noise
    noise_real = sigma * np.random.randn(total_samples)
    noise_imag = sigma * np.random.randn(total_samples)

    x_noisy = x + noise_real
    y_noisy = y + noise_imag

    # 4. Complex signal
    z_clean = x + 1j * y
    z_noisy = x_noisy + 1j * y_noisy

    noise_floor = np.sqrt(np.abs(z_noisy - z_clean) ** 2).mean()

    z = z_noisy

    # 5. Lagged dataset
    X, Y = [], []
    for i in range(lag, total_samples):
        X.append(z[i - lag:i])
        Y.append(z[i])

    X = np.array(X, dtype=np.complex64)
    Y = np.array(Y, dtype=np.complex64)

    # 6. PyTorch datasets
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 7. OOD (unchanged)
    if generate_ood:
        ood_samples = 500
        np.random.seed(seed + 1)

        X_rand = (np.random.randn(ood_samples, lag) +
                  1j * np.random.randn(ood_samples, lag)).astype(np.complex64)
        Y_rand = (np.random.randn(ood_samples) +
                  1j * np.random.randn(ood_samples)).astype(np.complex64)

        constant = np.random.randn(ood_samples) + 1j * np.random.randn(ood_samples)
        X_id = np.tile(constant[:, None], (1, lag)).astype(np.complex64)
        Y_id = constant.astype(np.complex64)

        X_ood = np.concatenate([X_rand, X_id], axis=0)
        Y_ood = np.concatenate([Y_rand, Y_id], axis=0)

        ood_dataset = TensorDataset(
            torch.from_numpy(X_ood),
            torch.from_numpy(Y_ood)
        )
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, ood_loader

    return train_loader, val_loader, test_loader

def generate_experiment_A(
        n_total: int = 15000,
        snr_db: float = 10.0,       # controls noise level
        batch_size: int = 32,
        seed: int = 42,
        generate_ood: bool = False
):
    np.random.seed(seed)

    def make_split(n, r_low, r_high, seed_offset=0):
        np.random.seed(seed + seed_offset)
        r   = np.random.uniform(r_low, r_high, n)
        phi = np.random.uniform(0, 2 * np.pi, n)

        # Clean signal
        z_clean = (r * np.exp(1j * phi)).astype(np.complex64)
        y_clean = r * np.sin(phi) + r**2 * np.cos(2 * phi)

        # Heteroscedastic noise shape (normalized)
        het_std = np.sqrt(0.1 * np.exp(r))
        het_std /= het_std.mean()          # normalize shape to unit mean

        # Scale noise shape to desired SNR
        signal_power = np.mean(y_clean ** 2)
        snr_linear   = 10 ** (snr_db / 10)
        noise_scale  = np.sqrt(signal_power / snr_linear)

        noise = het_std * noise_scale * np.random.randn(n)
        true_var = (het_std * noise_scale) ** 2

        Y = (y_clean + noise).astype(np.float32)

        return z_clean, Y, true_var.astype(np.float32)

    X, Y, true_var = make_split(n_total, 0.2, 0.8)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y)
    )
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    if generate_ood:
        X_ood, Y_ood, true_var_ood = make_split(len(test_dataset), 0.8, 1.2, seed_offset=1)
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_ood), torch.from_numpy(Y_ood), torch.from_numpy(true_var_ood)),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader, ood_loader

    return train_loader, test_loader, test_loader

if __name__ == "__main__":
    dataset = generate_sinusoidal_data()
    X,y = next(iter(dataset))
    print(X.shape, y.shape)
    visualize_complex_sequence(X, y)
