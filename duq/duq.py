import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


class DUQ(nn.Module):
    """
    Deterministic Uncertainty Quantification (DUQ) wrapper.
    Converts any CNN backbone into a DUQ model with uncertainty quantification.

    Args:
        backbone: A CNN model that extracts features (should output flattened features)
        feature_dim: Dimension of the features output by the backbone
        num_classes: Number of output classes
        embedding_size: Size of the embedding space for each class
        learnable_length_scale: Whether to learn the RBF kernel length scale
        length_scale: Initial value for the length scale (sigma)
        gamma: Exponential moving average decay for centroid updates
    """

    def __init__(
            self,
            backbone,
            feature_dim,
            num_classes,
            embedding_size,
            learnable_length_scale=False,
            length_scale=0.1,
            gamma=0.999,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.gamma = gamma

        # Projection matrix: maps features to embedding space per class
        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, feature_dim), 0.05)
        )

        # Running count of samples per class
        self.register_buffer("N", torch.ones(num_classes) * 12)

        # Running average of class centroids in embedding space
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )
        self.m = self.m * self.N.unsqueeze(0)

        # RBF kernel length scale
        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.register_buffer("sigma", torch.tensor(length_scale))

    def compute_features(self, x):
        """Extract features using the backbone model."""
        return self.backbone.compute_features(x)

    def last_layer(self, z):
        """
        Project features into embedding space for each class.

        Args:
            z: Features of shape [batch_size, feature_dim]

        Returns:
            Embeddings of shape [batch_size, embedding_size, num_classes]
        """
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        """
        Compute RBF kernel distances to class centroids.

        Args:
            z: Embeddings of shape [batch_size, embedding_size, num_classes]

        Returns:
            Distances/scores of shape [batch_size, num_classes]
        """
        # Normalize centroids by sample count
        embeddings = self.m / self.N.unsqueeze(0)

        # Compute squared distance to each class centroid
        diff = z - embeddings.unsqueeze(0)

        # Apply RBF kernel: exp(-||diff||^2 / (2 * sigma^2))
        distances = (-(diff ** 2)).mean(1).div(2 * self.sigma ** 2).exp()

        return distances


    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Class scores based on distance to centroids
        """
        z = self.compute_features(x)
        z = self.last_layer(z)
        y_pred = self.output_layer(z)
        return y_pred



    def initialize_centroids(self, train_loader, device):
        """Initialize centroids using actual training data."""
        print("Initializing centroids from data...")
        self.eval()

        class_embeddings = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for x, y in train_loader:
                x = x.float().to(device)
                y = y.long().to(device)

                # Get embeddings
                features = self.compute_features(x)
                z = self.last_layer(features)

                # Group by class
                for cls in range(self.num_classes):
                    mask = (y == cls)
                    if mask.any():
                        class_embeddings[cls].append(z[mask,:,cls])

            # Compute mean embeddings per class
            for cls in range(self.num_classes):
                if class_embeddings[cls]:
                    embeddings = torch.cat(class_embeddings[cls], dim=0)
                    self.m[:, cls] = embeddings.mean(dim=0)
                    self.N[cls] = len(embeddings)

        print("Centroids initialized!")
        self.train()

    def update_embeddings(self, x, y):
        """
        Update class centroids using exponential moving average.
        Should be called during training after each batch.

        Args:
            x: Input tensor
            y: One-hot encoded labels of shape [batch_size, num_classes]
        """
        with torch.no_grad():
            z = self.compute_features(x)
            z = self.last_layer(z)

            # Update sample count per class
            self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

            # Compute sum of embeddings per class
            features_sum = torch.einsum("ijk,ik->jk", z, y)

            # Update centroid running average
            self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def calc_gradient_penalty(self, x):
        """
        Calculate gradient penalty for Lipschitz regularization.
        Enforces smooth predictions by penalizing gradient norms that deviate from 1.

        Args:
            x: Input tensor (will be set to require gradients internally)

        Returns:
            Gradient penalty loss (scalar)
        """
        # Ensure input requires gradients
        x = x.clone().detach().requires_grad_(True)

        # Forward pass
        y_pred = self.forward(x)
        y_pred_sum = y_pred.sum()

        # Compute gradients of output w.r.t. input
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten gradients and compute L2 norm
        gradients = gradients.flatten(start_dim=1)
        grad_norm = gradients.norm(2, dim=1)

        # Two-sided penalty: penalize deviation from norm = 1
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def get_uncertainty(self, x):
        # scores = self.forward(x)
        distances = self.get_distances(x)
        max_distance = distances.max(dim=1).values

        # # Normalize scores to pseudo-probabilities
        # probs = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)
        #
        # predictions = scores.argmax(dim=1)
        # max_scores = scores.max(dim=1)[0]
        #
        # # Compute entropy as uncertainty measure
        # entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        return max_distance

    def train_duq(self, epochs, train_loader, val_loader, optimizer, device, lgp):
        # self.initialize_centroids(train_loader, device)
        self.train()

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            epoch_gp_loss = 0.0
            epoch_ce_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = batch
                x = x.float().to(device)
                y = y.long().to(device)

                # Forward pass
                y_pred = self(x)

                # DIAGNOSTIC: Print first batch info
                if batch_idx == 0 and epoch == 0:
                    print(f"\n=== DIAGNOSTICS ===")
                    print(f"Input shape: {x.shape}")
                    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
                    print(f"Labels shape: {y.shape}")
                    print(f"Labels unique: {y.unique()}")
                    print(f"Output shape: {y_pred.shape}")
                    print(f"Output range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
                    print(f"Output mean: {y_pred.mean():.4f}")
                    print(f"Output std: {y_pred.std():.4f}")
                    print(f"Sample output: {y_pred[0]}")
                    print(f"===================\n")
                # print("\n=== BEFORE GRADIENT PENALTY ===")
                # print(f"x: shape={x.shape}, device={x.device}")
                # print(f"x: min={x.min().item():.6f}, max={x.max().item():.6f}")
                # print(f"x: has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}")

                # Classification loss
                y_onehot = F.one_hot(y, self.num_classes).float()
                ce_loss = F.binary_cross_entropy(y_pred, y_onehot)

                # print("NaN in y_pred?", torch.isnan(y_pred).any().item())
                # print("Inf in y_pred?", torch.isinf(y_pred).any().item())
                # print("Min y_pred:", y_pred.min().item())
                # print("Max y_pred:", y_pred.max().item())
                # Gradient penalty

                if lgp != 0:
                    gp_loss = self.calc_gradient_penalty(x)

                # Total loss
                if lgp != 0:
                    loss = ce_loss + lgp * gp_loss
                else:
                    loss = ce_loss

                # Backward pass
                loss.backward()

                # DIAGNOSTIC: Check gradients
                if batch_idx == 0 and epoch == 0:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            print(f"{name}: grad_norm = {param.grad.norm():.6f}")
                        else:
                            print(f"{name}: NO GRADIENT!")

                optimizer.step()

                # Update embeddings
                with torch.no_grad():
                    self.update_embeddings(x.detach(), y_onehot)

                # Track metrics
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                #epoch_gp_loss += gp_loss.item()

                # Calculate accuracy
                predictions = y_pred.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            avg_ce_loss = epoch_ce_loss / len(train_loader)
            avg_gp_loss = epoch_gp_loss / len(train_loader)
            accuracy = 100 * correct / total

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            #print(f"  CE Loss: {avg_ce_loss:.4f} | GP Loss: {avg_gp_loss:.4f}")
            print(f"  Training Accuracy: {accuracy:.2f}%")
            print("-" * 60)

            self.val(val_loader, device)

        return {
            'final_loss': avg_loss,
            'final_accuracy': accuracy
        }

    def val(self, val_dataloader, device):
        self.eval()
        val_loss = 0.000
        val_acc = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (val_x, val_y) in enumerate(val_dataloader):
                val_x = val_x.float().to(device)
                val_y = val_y.long().to(device)
                val_out = self(val_x)
                y_onehot = F.one_hot(val_y, self.num_classes).float()
                ce_loss = F.binary_cross_entropy(val_out, y_onehot)

                # Total loss
                val_loss += ce_loss
                predictions = val_out.argmax(dim=1)
                correct += (predictions == val_y).sum().item()
                total += val_y.size(0)

            print(f"Validation Loss: {val_loss/total}")
            print(f"Validation Accuracy: {correct/total}")

    def test_duq(self, test_loader, device):
        self.eval()
        batch_size = test_loader.batch_size
        batch_predictions = []
        uncertainty = 0.00
        correct = 0
        all_max_distances = []
        correct_max_distances = []
        wrong_max_distances = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.float().to(device)
                y = y.long().to(device)

                out = self(x)
                batch_predictions.append(out.cpu())
                result = out.argmax(dim=1)
                max_distances = out.max(dim=1).values
                uncertainty += max_distances.mean().item()
                correct += (result == y).float().mean().item()

                # Separate correct and incorrect predictions
                correct_mask = (result == y)
                correct_max_distances.extend(max_distances[correct_mask].tolist())
                wrong_max_distances.extend(max_distances[~correct_mask].tolist())
                all_max_distances.extend(max_distances.tolist())

        print(f"Test accuracy: {100 * correct / len(test_loader):.2f}%")
        print(f"Uncertainty: {uncertainty / len(test_loader):.4f}")
        print(f"Avg max distance (correct): {sum(correct_max_distances) / len(correct_max_distances):.4f}")
        print(
            f"Avg max distance (wrong): {sum(wrong_max_distances) / len(wrong_max_distances) if wrong_max_distances else 0:.4f}")

        return all_max_distances, correct_max_distances, wrong_max_distances









