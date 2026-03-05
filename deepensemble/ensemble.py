import numpy as np
from sklearn.metrics import accuracy_score
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from complexPytorch import CrossEntropyComplex, softmax_real_with_abs, softmax_real_with_avg, CrossEntropyComplexTwice, \
    softmax_complex
from deepensemble.uncertainty import calculate_classification_uncertainty, calculate_regression_uncertainty


# from CDSCNN.train import train_one_epoch
# from CDSCNN.validation import val

# from test import test_models


class DeepEnsemble:
    def __init__(self, model, num, model_kwargs, task = "classification",device='cuda'):
        self.device = device
        self.num = num
        self.models = []
        self.task = task.lower()

        if task not in ["classification", "regression"]:
            raise ValueError(f"task {task} is not supported.")

        for i in range(num):
            torch.manual_seed(i*100)
            m = model(**model_kwargs).to(self.device)
            self.models.append(m)


        self.backbone = self.models[0]

    def val(self, net, device, val_dataloader, loss_func, val_loss_history, val_metric_history):
        net.eval()
        total_loss = 0.0
        total_metric = 0.0

        with torch.no_grad():
            for val_x, val_y in val_dataloader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                if self.task == 'classification':
                    out = net(val_x)
                    loss = loss_func(out, val_y)
                    total_metric += (out.argmax(dim=1) == val_y).float().mean().item()
                else:
                    mean, var = net(val_x)
                    loss = loss_func(mean, var, val_y)
                    # .abs() required for complex-valued outputs (squared modulus)
                    total_metric += torch.sqrt((torch.abs(mean - val_y) ** 2).mean()).item()

                total_loss += loss.item()

        num_batches = len(val_dataloader)
        val_loss_history.append(total_loss / num_batches)
        val_metric_history.append(total_metric / num_batches)

    def train_ensemble(self, train_dataloader, val_dataloader, epochs, lr=0.001, loss_func=torch.nn.CrossEntropyLoss()):
        for i, net in enumerate(self.models):
            train_loss_history = []
            train_metric_history = []
            val_loss_history = []
            val_metric_history = []

            print(f"Training model {i + 1} of {len(self.models)}")

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            # Bootstrap sample (with replacement) for ensemble diversity
            train_dataset = train_dataloader.dataset
            indices = torch.randint(0, len(train_dataset), (len(train_dataset),))
            bootstrapped_dataset = torch.utils.data.Subset(train_dataset, indices)
            model_train_loader = DataLoader(
                bootstrapped_dataset,
                batch_size=train_dataloader.batch_size,
                shuffle=True
            )

            for epoch in range(epochs):
                net.train()
                total_loss = 0.0
                total_metric = 0.0
                num_batches = 0

                for x, y in model_train_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    optimizer.zero_grad()

                    if self.task == 'classification':
                        out = net(x)
                        l = loss_func(out, y)
                    else:
                        mean, var = net(x)
                        l = loss_func(mean, var, y)

                    l.backward()
                    optimizer.step()

                    total_loss += l.item()
                    num_batches += 1

                    with torch.no_grad():
                        if self.task == 'classification':
                            total_metric += (out.argmax(dim=1) == y).float().mean().item()
                        else:
                            total_metric += torch.sqrt((torch.abs(mean - y) ** 2).mean()).item()

                avg_loss = total_loss / num_batches
                avg_metric = total_metric / num_batches

                train_loss_history.append(avg_loss)
                train_metric_history.append(avg_metric)

                self.val(net, self.device, val_dataloader, loss_func, val_loss_history, val_metric_history)

                metric_label = "Accuracy" if self.task == 'classification' else "RMSE"
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train {metric_label}: {avg_metric:.4f} | Train Loss: {avg_loss:.4f} | "
                      f"Val {metric_label}: {val_metric_history[-1]:.4f} | Val Loss: {val_loss_history[-1]:.4f}")

            torch.cuda.empty_cache()


    def test_ensemble(self, test_dataloader, temp_scaler = None):

        for model in self.models:
            model.eval()

        num_models = len(self.models)
        all_ensemble_predictions = []  # Will store predictions for all batches
        all_results = []  # Will store all true labels
        all_mean = []
        all_var_real = []
        all_var_imag = []
        test_loss = 0.0

        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Collect predictions from all models for this batch
                batch_predictions = []
                batch_mean = []
                batch_var_real = []
                batch_var_imag = []
                for model in self.models:
                    if self.task == 'classification':
                        logits = model(x)
                        if temp_scaler:
                            logits = temp_scaler(logits)

                        batch_predictions.append(logits.cpu())
                    else:
                        mean, var = model(x)
                        batch_predictions.append(mean.cpu())
                        batch_var_real.append(var[0].cpu())
                        batch_var_imag.append(var[1].cpu())

                    # probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities


                # Stack: (num_models, batch_size, num_classes)
                batch_predictions = torch.stack(batch_predictions, dim=0)
                # batch_mean = torch.stack(batch_mean, dim=0)
                batch_var_real = torch.stack(batch_var_real, dim=0)
                batch_var_imag = torch.stack(batch_var_imag, dim=0)

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
                all_results.append(y.cpu())
                # all_mean.append(batch_mean)
                all_var_real.append(batch_var_real)
                all_var_imag.append(batch_var_imag)

        # Concatenate all batches
        # Shape: (num_models, total_samples, num_classes)
        ensemble_predictions = torch.cat(all_ensemble_predictions, dim=1).numpy()
        # ensemble_mean = torch.cat(all_mean, dim=1).numpy()
        ensemble_var_real = torch.cat(all_var_real, dim=1).numpy()
        ensemble_var_imag = torch.cat(all_var_imag, dim=1).numpy()
        true_results = torch.cat(all_results, dim=0).numpy()

        # Calculate comprehensive uncertainty metrics
        if self.task == "classification":
            metrics = calculate_classification_uncertainty(ensemble_predictions, true_results)
            print(f"Expected ID entropy: {metrics['expected_entropy_id']}")

            # Print summary
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            # print(f"Test Loss: {metrics['test_loss']:.4f}")
            print(f"Test NLL: {metrics['nll']:.4f}")
            print(f"Brier Score: {metrics['brier_score']:.4f}")
            print(f"Avg Predictive Entropy: {metrics['predictive_entropy']:.4f}")
            print(f"Avg Mutual Information: {metrics['mutual_information']:.4f}")
            print(f"Avg Prediction Variance: {metrics['avg_prediction_variance']:.4f}")
            print(f"ECE: {metrics['ece']:.4f}")

        else:
            metrics = calculate_regression_uncertainty(ensemble_predictions, ensemble_var_real, ensemble_var_imag, true_results)

        return metrics

    def forward(self, x):
        logits = []
        for model in self.models:
            out = model(x)  # logits
            logits.append(out)

        logits = torch.stack(logits, dim=0)  # [num_models, B, C]
        mean_logits = logits.mean(dim=0)  # [B, C]

        return mean_logits








