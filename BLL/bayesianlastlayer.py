import numpy as np
import torch
import vbll
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from heatmap import plot_class_prob_heatmap, compute_ece


# from heatmap import plot_uncertainty_heatmap, plot_uncertainty_surface


class BLLModel(nn.Module):
    def __init__(self, backbone, num_classes: int, feature_dim: int, model_kwargs, device,
                 train_dataloader: DataLoader):
        super(BLLModel, self).__init__()
        self.backbone = backbone(**model_kwargs).to(device)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.bll_layer = vbll.DiscClassification(self.feature_dim, self.num_classes,
                                                0.5 / train_dataloader.__len__()).to(device)
        self.device = device

    def forward(self, x):
        x = self.backbone.compute_features(x)
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

    def test_model(self, dataloader, is_ood=False, n_samples=1):
        self.eval()

        test_loss = 0.0
        test_acc = 0.0
        total_nll = 0.0

        probs_all = []
        preds_all = []
        labels_all = []
        max_probs_all = []
        entropy_all = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # CRITICAL FIX: Sample multiple times from posterior
                sampled_probs = []
                for _ in range(n_samples):
                    out = self(x)
                    sampled_probs.append(out.predictive.probs)

                # Average predictions across samples
                sampled_probs = torch.stack(sampled_probs)  # (n_samples, B, C)
                probs = sampled_probs.mean(dim=0)  # (B, C) - predictive mean

                # Compute epistemic uncertainty (optional but useful)
                epistemic_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                aleatoric_entropy = (-(sampled_probs * torch.log(sampled_probs + 1e-10)).sum(dim=2)).mean(dim=0)
                total_entropy = epistemic_entropy

                max_probs, preds = torch.max(probs, dim=1)

                probs_all.append(probs.cpu())
                max_probs_all.append(max_probs.cpu())
                entropy_all.append(total_entropy.cpu())
                preds_all.append(preds.cpu())
                labels_all.append(y.cpu())

            probs_all = torch.cat(probs_all)
            max_probs_all = torch.cat(max_probs_all)
            entropy_all = torch.cat(entropy_all)
            preds_all = torch.cat(preds_all)
            labels_all = torch.cat(labels_all)


        if not is_ood:
            # preds_all = torch.cat(preds_all)
            # labels_all = torch.cat(labels_all)

            correct_mask = preds_all == labels_all
            incorrect_mask = ~correct_mask

            avg_max_correct = max_probs_all[correct_mask].mean().item()
            avg_max_incorrect = max_probs_all[incorrect_mask].mean().item()

            test_acc = correct_mask.float().mean().item()

            # NLL from predictive mean
            true_probs = probs_all[torch.arange(len(labels_all)), labels_all]
            avg_nll = -torch.log(true_probs + 1e-10).mean().item()

            # test_loss /= len(dataloader)
            # test_acc /= len(dataloader)
            # avg_nll = total_nll / len(dataloader)
            ece = compute_ece(probs_all.numpy(), preds_all.numpy(), labels_all.numpy())

            print("=== ID Evaluation ===")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Average NLL: {avg_nll:.4f}")
            print(f"ECE: {ece:.4f}")
            # print(f"Avg entropy (ID): {entropy_all.mean().item():.4f}")
            print(f"Avg max prob (correct):   {avg_max_correct:.4f}")
            print(f"Avg max prob (incorrect): {avg_max_incorrect:.4f}")

            return preds_all,entropy_all,max_probs_all,labels_all


        else:
            print("=== OOD Evaluation ===")
            print(f"Avg entropy (OOD): {entropy_all.mean().item():.4f}")
            print(f"Avg max softmax prob (OOD): {max_probs_all.mean().item():.4f}")

            return preds_all,entropy_all,max_probs_all



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

