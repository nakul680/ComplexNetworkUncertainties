import numpy as np
from sklearn.metrics import accuracy_score
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from complexPytorch import CrossEntropyComplex, softmax_real_with_abs, softmax_real_with_avg, CrossEntropyComplexTwice, \
    softmax_complex
from deepensemble.uncertainty import calculate_classification_uncertainty
# from CDSCNN.train import train_one_epoch
# from CDSCNN.validation import val

# from test import test_models


class DeepEnsemble:
    def __init__(self, model, num, model_kwargs, device='cuda'):
        self.device = device
        self.num = num
        self.models = []

        for i in range(num):
            torch.manual_seed(i*100)
            m = model(**model_kwargs).to(self.device)
            self.models.append(m)


    def val(self, net, device, val_dataloader, loss_func, val_LOSS, val_AC):

        net.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.float().to(device)
                val_y = val_y.long().to(device)

                outputs = net(val_x)
                loss = loss_func(outputs, val_y)

                total_loss += loss.item() * val_x.size(0)

                # if self.models[0].is_complex:
                #     outputs = softmax_real_with_avg(outputs, dim=1)
                # else:
                #     outputs = torch.softmax(outputs, dim=1)

                preds = outputs.argmax(dim=1)
                correct += (preds == val_y).sum().item()
                total += val_y.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total

        val_LOSS.append(avg_loss)
        val_AC.append(avg_acc)

    def train_ensemble(self, train_dataloader, val_dataloader, epochs, lr = 0.001, loss_func = torch.nn.CrossEntropyLoss()):
        is_complex = self.models[0].is_complex
        for i in range(len(self.models)):
            net = self.models[i]
            loss = []
            acc = []
            val_loss = []
            val_acc = []
            print(f"Training model {i+1} of {len(self.models)}")
            # torch.manual_seed(i)
            optimizer = torch.optim.Adam(self.models[i].parameters(), lr=lr)

            # Create model-specific dataloaders with different shuffle seeds
            train_dataset = train_dataloader.dataset
            indices = torch.randint(0, len(train_dataset), (len(train_dataset),))
            bootstrapped_dataset = torch.utils.data.Subset(train_dataset, indices)

            model_train_loader = DataLoader(
                bootstrapped_dataset,
                batch_size=train_dataloader.batch_size,
                shuffle=True
            )

            for epoch in range(epochs):
                # torch.manual_seed(i*epoch)
                net.train()
                Loss = 0
                ac = 0
                for _,(x, y) in enumerate(model_train_loader):
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    optimizer.zero_grad()
                    out = net(x)
                    l = loss_func(out, y)
                    Loss += l.item()
                    # if is_complex:
                    #     l.abs().backward()
                    # else:
                    l.backward()
                    optimizer.step()
                    # if is_complex:
                    #     out = softmax_real_with_avg(out, dim=1)
                    # else:
                    #     out = torch.softmax(out, dim=1)

                    ac += accuracy_score(y.cpu().detach().numpy(), torch.max(out, 1)[1].cpu().detach().numpy())

                # if is_complex:
                #     Loss = abs(Loss)

                loss.append(Loss / len(model_train_loader))
                acc.append(ac / len(model_train_loader))
                self.val(self.models[i], self.device, val_dataloader, loss_func, val_loss, val_acc)
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Accuracy: {acc[epoch]}")
                print(f"Train Loss: {loss[epoch]}")
                print(f"Valid Accuracy: {val_acc[epoch]}")
                print(f"Valid Loss: {val_loss[epoch]}")

            torch.cuda.empty_cache()


    def test_ensemble(self, test_dataloader,loss_func, ood = False):

        for model in self.models:
            model.eval()

        num_models = len(self.models)
        all_ensemble_predictions = []  # Will store predictions for all batches
        all_labels = []  # Will store all true labels
        test_loss = 0.0

        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # Collect predictions from all models for this batch
                batch_predictions = []
                for model in self.models:
                    # logits = model(x)
                    # if model.is_complex:
                    #     probs = softmax_real_with_avg(logits, dim=1)
                    # else:
                    #     probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
                    logits = model(x)
                    # probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
                    batch_predictions.append(logits.cpu())

                # Stack: (num_models, batch_size, num_classes)
                batch_predictions = torch.stack(batch_predictions, dim=0)

                # Calculate mean prediction for loss
                mean_pred = batch_predictions.mean(dim=0).to(self.device)
                # if not ood:
                #     loss = loss_func(mean_pred, y)
                #     test_loss += loss.item()
                #
                # else:
                #     test_loss = 0

                # Store for later uncertainty calculation
                all_ensemble_predictions.append(batch_predictions)
                all_labels.append(y.cpu())

        # Concatenate all batches
        # Shape: (num_models, total_samples, num_classes)
        ensemble_predictions = torch.cat(all_ensemble_predictions, dim=1).numpy()
        true_labels = torch.cat(all_labels, dim=0).numpy()

        # Calculate comprehensive uncertainty metrics

        metrics = calculate_classification_uncertainty(ensemble_predictions, true_labels)
        print(f"Expected ID entropy: {metrics['expected_entropy_id']}")

        # Add loss to metrics
        # metrics['test_loss'] = test_loss / len(test_dataloader)

        # Calculate additional uncertainty metrics
        # epsilon = 1e-10
        #
        # # Mutual Information
        # mean_predictions = ensemble_predictions.mean(axis=0)
        # # pred_entropy = -np.sum(mean_predictions * np.log(mean_predictions + epsilon), axis=1)
        # pred_entropy = -np.sum(mean_predictions * mean_predictions, axis=1)
        # # individual_entropy = -np.sum(
        # #     ensemble_predictions * np.log(ensemble_predictions + epsilon),
        # #     axis=2
        # # )
        # individual_entropy = -np.sum(
        #         ensemble_predictions * ensemble_predictions ,
        #         axis=2
        #     )
        # exp_entropy = individual_entropy.mean(axis=0)
        # mutual_info = pred_entropy - exp_entropy


        # metrics['mutual_information'] = mutual_info.mean()
        # metrics['mi_per_sample'] = mutual_info
        # metrics['expected_entropy'] = exp_entropy.mean()
        #
        # # Prediction variance (epistemic uncertainty)
        # prediction_variance = ensemble_predictions.var(axis=0)
        # metrics['avg_prediction_variance'] = prediction_variance.mean()
        # metrics['variance_per_sample'] = prediction_variance.mean(axis=1)

        # Print summary
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        # print(f"Test Loss: {metrics['test_loss']:.4f}")
        print(f"Test NLL: {metrics['nll']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Avg Predictive Entropy: {metrics['predictive_entropy']:.4f}")
        print(f"Avg Mutual Information: {metrics['mutual_information']:.4f}")
        print(f"Avg Prediction Variance: {metrics['avg_prediction_variance']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")

        return metrics

#         predictions = torch.cat(predictions,dim=0)
#         mean = torch.mean(predictions, dim=0)
#         #std = torch.std(predictions, dim=1)
#         #mean_std = torch.mean(std, dim=0)
#
#         print(f"Mean of Predictions: {mean}")
#         print(f"Std of Predictions: {mean}")





