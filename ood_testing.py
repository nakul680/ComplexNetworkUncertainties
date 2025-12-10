import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import pickle

from bll_pipeline import bll_experiment
from complexnetwork.complexCNN import ComplexCNN
from realnetwork.amc_cnn import AMC_CNN

if __name__ == "__main__":
    # Load data
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()

    snrs = sorted(list(set([key[1] for key in p.keys()])))
    mods = sorted(list(set([key[0] for key in p.keys()])))
    num_classes = len(mods)

    print("All Classes:", mods)

    # ---------------------------------------------------
    # 1. SELECT ONE RANDOM CLASS AS OOD
    # ---------------------------------------------------
    np.random.seed(2016293)
    ood_class_idx = np.random.randint(0, len(mods))
    ood_class = mods[ood_class_idx]

    print(f"\nOOD Class (removed from train/val): {ood_class}")

    # Separate in-distribution and OOD classes
    id_mods = [mod for mod in mods if mod != ood_class]
    print(f"In-Distribution Classes ({len(id_mods)}): {id_mods}")

    # ---------------------------------------------------
    # 2. BUILD DATASETS SEPARATELY
    # ---------------------------------------------------
    # In-distribution data (for train/val/test)
    X_id_list = []
    y_id_list = []

    for mod in id_mods:
        for snr in snrs:
            samples = p[(mod, snr)]  # shape: [N, 2, 128]
            X_id_list.append(samples)
            # Map to new class indices (0 to len(id_mods)-1)
            y_id_list += [id_mods.index(mod)] * samples.shape[0]

    X_id = np.vstack(X_id_list)
    Y_id = np.array(y_id_list)
    N_id = len(Y_id)

    # OOD data
    X_ood_list = []
    y_ood_list = []

    for snr in snrs:
        samples = p[(ood_class, snr)]
        X_ood_list.append(samples)
        # Keep original class index or use a special OOD label
        y_ood_list += [100] * samples.shape[0]

    X_ood = np.vstack(X_ood_list)
    Y_ood = np.array(y_ood_list)
    N_ood = len(Y_ood)

    print(f"\nIn-Distribution samples: {N_id}")
    print(f"OOD samples: {N_ood}")

    # ---------------------------------------------------
    # 3. TRAIN/VAL/TEST SPLIT (ID data only)
    # ---------------------------------------------------
    indices_id = np.arange(N_id)
    np.random.shuffle(indices_id)

    n_train = int(0.6 * N_id)
    n_valid = int(0.2 * N_id)
    n_test = N_id - n_train - n_valid

    train_idx = indices_id[:n_train]
    valid_idx = indices_id[n_train:n_train + n_valid]
    test_idx = indices_id[n_train + n_valid:]

    X_train = X_id[train_idx]
    X_valid = X_id[valid_idx]
    X_test = X_id[test_idx]

    Y_train = Y_id[train_idx]
    Y_valid = Y_id[valid_idx]
    Y_test = Y_id[test_idx]

    print(f"\nTrain samples: {len(Y_train)}")
    print(f"Valid samples: {len(Y_valid)}")
    print(f"Test samples: {len(Y_test)}")

    # ---------------------------------------------------
    # 4. CONVERT TO TORCH TENSORS
    # ---------------------------------------------------
    X_train_tensor = torch.from_numpy(X_train).float()
    X_valid_tensor = torch.from_numpy(X_valid).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    X_ood_tensor = torch.from_numpy(X_ood).float()

    Y_train_tensor = torch.from_numpy(Y_train).long()
    Y_valid_tensor = torch.from_numpy(Y_valid).long()
    Y_test_tensor = torch.from_numpy(Y_test).long()
    Y_ood_tensor = torch.from_numpy(Y_ood).long()

    # ---------------------------------------------------
    # 5. CREATE DATALOADERS
    # ---------------------------------------------------
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    ood_dataset = TensorDataset(X_ood_tensor, Y_ood_tensor)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=100, shuffle=False)

    print("\n✓ DataLoaders created successfully!")
    print(f"  - train_loader: {len(train_loader)} batches")
    print(f"  - valid_loader: {len(valid_loader)} batches")
    print(f"  - test_loader: {len(test_loader)} batches")
    print(f"  - ood_loader: {len(ood_loader)} batches (OOD class: {ood_class})")

    # ---------------------------------------------------
    # 6. VERIFICATION
    # ---------------------------------------------------
    print("\n--- Class Distribution Verification ---")
    print(f"Train classes: {np.unique(Y_train)} (count: {len(np.unique(Y_train))})")
    print(f"Valid classes: {np.unique(Y_valid)} (count: {len(np.unique(Y_valid))})")
    print(f"Test classes: {np.unique(Y_test)} (count: {len(np.unique(Y_test))})")
    print(f"OOD classes: {np.unique(Y_ood)} (count: {len(np.unique(Y_ood))})")
    #
    # test_dataset = ConcatDataset([test_dataset, ood_dataset])
    # test_loader = DataLoader(test_dataset, batch_size=110, shuffle=False)

    bll_experiment(AMC_CNN, ComplexCNN, train_loader, valid_loader, test_loader, len(id_mods), 1, 0.0001, ood = True, ood_dataloader=ood_loader)







