import matplotlib.pyplot as plt
import umap.umap_ as umap
import numpy as np


def plot_umap(
    id_emb, id_labels,
    ood_emb, centroids_emb,
    title,
    subsample_rate=200,
    seed=42
):
    np.random.seed(seed)

    # ---- Subsample ID ----
    id_idx = np.arange(len(id_emb))
    id_keep = id_idx[::subsample_rate]

    id_emb_sub = id_emb[id_keep]
    id_labels_sub = id_labels[id_keep]

    # ---- Subsample OOD ----
    ood_idx = np.arange(len(ood_emb))
    ood_keep = ood_idx[::subsample_rate]

    ood_emb_sub = ood_emb[ood_keep]

    # ---- Plot ----
    plt.figure(figsize=(9, 7))

    # ID samples
    plt.scatter(
        id_emb_sub[:, 0], id_emb_sub[:, 1],
        c=id_labels_sub,
        cmap="tab10",
        s=15, alpha=0.7,
        label="ID"
    )

    # OOD samples
    plt.scatter(
        ood_emb_sub[:, 0], ood_emb_sub[:, 1],
        c="black",
        s=15, alpha=0.4,
        label="OOD"
    )

    # Centroids (no subsampling!)
    plt.scatter(
        centroids_emb[:, 0], centroids_emb[:, 1],
        c="red",
        s=220,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Centroids"
    )

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def run_umap(arrays):
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )

    stacked = np.vstack(arrays)
    embedding = reducer.fit_transform(stacked)

    splits = np.cumsum([len(a) for a in arrays])
    return np.split(embedding, splits[:-1])
