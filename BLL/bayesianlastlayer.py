import numpy as np
import torch
import vbll
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from heatmap import plot_class_prob_heatmap


# from heatmap import plot_uncertainty_heatmap, plot_uncertainty_surface


class BLLModel(nn.Module):
    def __init__(self, backbone, num_classes: int, model_kwargs, device, train_dataloader: DataLoader):
        super(BLLModel, self).__init__()
        self.backbone = backbone(**model_kwargs).to(device)
        self.num_classes = num_classes
        self.bll_layer = vbll.DiscClassification(self.num_classes, self.num_classes,
                                                 1. / train_dataloader.__len__()).to(device)
        self.device = device

    def forward(self, x):
        x = self.backbone(x)
        x = self.bll_layer(x)
        return x

    def train_one_epoch(self, train_dataloader, optimizer, LOSS, AC, nll):
        self.train()
        loss = 0
        acc = 0
        nll_epoch = 0
        for i, (x, y) in enumerate(train_dataloader):
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            optimizer.zero_grad()
            #        x = x.permute(0, 2, 1)  # swap last two dimensions
            out = self.forward(x)
            probs = out.predictive.probs
            # print(out.predictive.probs[0])
            # out = out.predictive.logits
            l = out.train_loss_fn(y)
            # loss = loss_func(out, y)
            loss += l.item()
            l.backward()
            optimizer.step()
            true_probs = probs[torch.arange(len(y)), y]
            nll_epoch += -torch.log(true_probs + 1e-10).mean().item()
            preds = out.predictive.probs.argmax(dim=-1)  # if classification
            acc += (preds == y).float().mean().item()

        LOSS.append(loss / len(train_dataloader))
        AC.append(acc / len(train_dataloader))
        nll.append(nll_epoch / len(train_dataloader))

    def train_model(self, epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer, logs):
        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []
        nll = []
        num_batches = len(train_dataloader)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.train_one_epoch(train_dataloader, optimizer, losses, accuracies, nll)
            print(f"Loss: {losses[epoch]}")
            print(f"Accuracy: {accuracies[epoch]}")
            self.val(val_dataloader, val_losses, val_accuracies)
            print(f"Val Loss: {val_losses[epoch]}")
            print(f"Val Accuracy: {val_accuracies[epoch]}")
            print(f"NLL: {nll[epoch]}")

        logs["loss"] = losses
        logs["accuracy"] = accuracies
        logs["val_loss"] = val_losses
        logs["val_accuracy"] = val_accuracies
        logs["nll"] = nll

    def val(self, val_dataloader, val_LOSS, val_AC):
        self.eval()
        val_loss = 0.000
        val_acc = 0
        with torch.no_grad():
            for i, (val_x, val_y) in enumerate(val_dataloader):
                val_x = val_x.float().to(self.device)
                val_y = val_y.long().to(self.device)
                val_out = self(val_x)
                val_l = val_out.train_loss_fn(val_y)
                val_loss += val_l.item()
                preds = val_out.predictive.probs.argmax(dim=-1)
                val_acc += (preds == val_y).float().mean().item()

            val_LOSS.append(val_loss / (i + 1))
            val_AC.append(val_acc / (i + 1))

    def test_model(self, test_dataloader):
        self.eval()
        test_loss = 0.000
        test_acc = 0
        total_nll = 0.00
        batch_size = test_dataloader.batch_size
        pred = []
        nll = []
        labels = []
        entropy = []
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.float().to(self.device)
                y = y.long().to(self.device)


                out = self(x)
                labels.extend(y.cpu())
                batch_preds = out.predictive.probs
                pred.extend(batch_preds.cpu().numpy())
                predictive_entropy = -(batch_preds * torch.log(batch_preds + 1e-10)).sum(dim=1)
                true_class_probs = batch_preds[torch.arange(batch_size), y]
                total_nll += -torch.log(true_class_probs + 1e-10).mean()
                nll.append(-torch.log(true_class_probs + 1e-10).mean())
                loss = out.train_loss_fn(y)
                test_loss += loss.item()
                preds = batch_preds.argmax(dim=-1)
                test_acc += (preds == y).float().mean().item()
                entropy.extend(predictive_entropy.cpu().numpy())

        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
        ece = self.compute_ece(pred, labels)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Average NLL:{total_nll / test_dataloader.__len__():.4f}")
        print(f"ECE: {ece:.4f}")
        print(f"Avg entropy for in distribution data:{np.mean(entropy):.4f}")

        #self.get_dist(x,y)
        prob_matrix = plot_class_prob_heatmap(self, test_dataloader, self.num_classes, self.device)
        # fig, axes = plt.subplots(4, 3, figsize=(12, 12))
        # axes = axes.flatten()
        #
        # for i, (arr, ax) in enumerate(zip(prob_matrix, axes)):
        #     colors = ['skyblue'] * len(batch_preds)  # default color for all
        #     colors[i] = 'orange'
        #     ax.bar(range(len(arr)), arr, color=colors)
        #     ax.set_ylim(0.00, 1.00)
        #     ax.set_title(f"Class {i} predictions")
        #
        # axes[-1].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        return prob_matrix

    def get_dist(self, x, y):
        self.eval()
        # random_batch = np.random.choice(list(test_dataloader))
        # x, y = random_batch
        # x = x.float().to(self.device)
        # y = y.long().to(self.device)
        batch_preds = []
        highlight_index = y[0]
        for _ in range(10):
            batch_preds.append(self(x).predictive.probs[0].detach().cpu())

        batch_preds = torch.stack(batch_preds)
        batch_preds = batch_preds.mean(dim=0)

        colors = ['skyblue'] * len(batch_preds)  # default color for all
        colors[highlight_index] = 'orange'

        plt.bar(range(len(batch_preds)), batch_preds, color=colors)
        plt.xlabel('Bins')
        plt.ylabel('Value')
        plt.title('Bar Plot with One Highlighted Bin')
        plt.show()

    def compute_ece(self,probs, labels, n_bins=15):
        """
        Compute Expected Calibration Error (ECE).

        Parameters
        ----------
        probs : array-like, shape (N,) or (N, C)
            Predicted probabilities. For multiclass, pass softmax probabilities.
        labels : array-like, shape (N,)
            True labels (class indices for multiclass, or 0/1 for binary).
        n_bins : int
            Number of bins for calibration calculation.

        Returns
        -------
        float
            Expected Calibration Error (ECE)
        """

        probs = np.array(probs)
        labels = np.array(labels)

        # If multiclass: use the predicted class probability for each sample
        if probs.ndim == 2:
            # Select probability assigned to the true class
            probs = probs[np.arange(len(probs)), labels]

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1

        ece = 0.0
        N = len(probs)

        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_probs = probs[mask]
                bin_labels = labels[mask]

                # Confidence is mean predicted probability
                conf = np.mean(bin_probs)

                # Accuracy is fraction of correct labels
                acc = np.mean(bin_labels == (bin_probs >= 0.5)) if probs.ndim == 1 else np.mean(
                    bin_labels == labels[mask])

                # Weighted difference
                ece += (len(bin_probs) / N) * abs(acc - conf)

        return ece


    def test_ood(self, ood_loader):
        self.eval()
        batch_size = ood_loader.batch_size
        pred = []
        labels = []
        entropy = []
        with torch.no_grad():
            for x, y in ood_loader:
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                out = self(x)
                labels.extend(y.cpu())
                batch_preds = out.predictive.probs.cpu().numpy()
                predictive_entropy = -np.sum(batch_preds * np.log(batch_preds + 1e-10), axis=1)
                # avg_entropy = predictive_entropy.mean()
                pred.extend(batch_preds)
                entropy.extend(predictive_entropy)



        print(f"Avg entropy for ood:{np.mean(entropy):.4f}")





