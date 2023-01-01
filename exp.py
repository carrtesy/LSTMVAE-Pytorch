import torch
import torch.nn.functional as F
import numpy as np
import copy
import os
import wandb
from model import LSTMVAE

from tqdm import tqdm
import pickle
from utils.metrics import PA
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score

class LSTMVAE_Trainer:
    def __init__(self, args, logger, train_loader, test_loader):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = LSTMVAE(
            input_dim=self.args.num_channels,
            hidden_dim=self.args.hidden_dim,
            z_dim=self.args.z_dim,
            n_layers=self.args.n_layers,
        ).to(self.args.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)

    @staticmethod
    def gaussian_prior_KLD(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def train(self):
        wandb.watch(self.model, log="all", log_freq=100)

        train_iterator = tqdm(
            range(1, self.args.epochs + 1),
            total=self.args.epochs,
            desc="training epochs",
            leave=True
        )

        best_train_stats = None
        for epoch in train_iterator:
            train_stats = self.train_epoch()
            self.logger.info(f"epoch {epoch} | train_stats: {train_stats}")
            self.checkpoint(os.path.join(self.args.checkpoint_path, f"epoch{epoch}.pth"))

            if best_train_stats is None or train_stats < best_train_stats:
                self.logger.info(f"Saving best results @epoch{epoch}")
                self.checkpoint(os.path.join(self.args.checkpoint_path, f"best.pth"))
                best_train_stats = train_stats
        return

    def train_epoch(self):
        self.model.train()
        log_freq = len(self.train_loader) // self.args.log_freq
        train_summary = 0.0
        for i, batch_data in enumerate(self.train_loader):
            train_log = self._process_batch(batch_data)
            if (i + 1) % log_freq == 0:
                self.logger.info(f"{train_log}")
                wandb.log(train_log)
            train_summary += train_log["summary"]
        train_summary /= len(self.train_loader)
        return train_summary

    def _process_batch(self, batch_data) -> dict:
        X = batch_data[0].to(self.args.device)
        B, L, C = X.shape

        Xhat, mu, logvar = self.model(X)

        self.optimizer.zero_grad()
        recon_loss = F.mse_loss(Xhat, X)
        KLD_loss = self.gaussian_prior_KLD(mu, logvar)
        loss = recon_loss + self.args.beta * KLD_loss
        loss.backward()
        self.optimizer.step()

        out = {
            "recon_loss": recon_loss.item(),
            "KLD_loss": KLD_loss.item(),
            "total_loss": loss.item(),
            "summary": loss.item(),
        }
        return out

    @torch.no_grad()
    def infer(self):
        result = {}
        self.model.eval()
        gt = self.test_loader.dataset.y
        recon_errors = self.calculate_recon_errors()
        anomaly_scores = self.reduce(recon_errors)

        # thresholding
        threshold = self.get_threshold(gt=gt, anomaly_scores=anomaly_scores)
        result.update({"Threshold": threshold})

        # AUROC
        s = anomaly_scores - threshold
        logit = 1 / (1 + np.exp(-s))  # (N, )
        pred_prob = np.zeros((len(logit), 2))
        pred_prob[:, 0], pred_prob[:, 1] = 1 - logit, logit
        wandb.sklearn.plot_roc(gt, pred_prob)
        auc = roc_auc_score(gt, anomaly_scores)
        result.update({"AUC": auc})

        # F1
        pred = (anomaly_scores > threshold).astype(int)
        acc = accuracy_score(gt, pred)
        p = precision_score(gt, pred, zero_division=1)
        r = recall_score(gt, pred, zero_division=1)
        f1 = f1_score(gt, pred, zero_division=1)

        result.update({
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1": f1,
        })
        wandb.sklearn.plot_confusion_matrix(gt, pred, labels=["normal", "abnormal"])

        # F1-PA
        pa_pred = PA(gt, pred)
        acc = accuracy_score(gt, pa_pred)
        p = precision_score(gt, pa_pred, zero_division=1)
        r = recall_score(gt, pa_pred, zero_division=1)
        f1 = f1_score(gt, pa_pred, zero_division=1)
        result.update({
            "Accuracy (PA)": acc,
            "Precision (PA)": p,
            "Recall (PA)": r,
            "F1 (PA)": f1,
        })
        wandb.sklearn.plot_confusion_matrix(gt, pa_pred, labels=["normal", "abnormal"])
        return result

    def calculate_recon_errors(self):
        eval_iterator = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="calculating reconstruction errors",
            leave=True
        )

        recon_errors = []
        for i, batch_data in enumerate(eval_iterator):
            X = batch_data[0].to(self.args.device)
            B, L, C = X.shape
            Xhat, _, _ = self.model(X)
            recon_error = F.mse_loss(Xhat, X, reduction='none').to("cpu")
            recon_errors.append(recon_error)
        recon_errors = np.concatenate(recon_errors, axis=0)
        return recon_errors

    @staticmethod
    def reduce(arr, stride=1):
        '''
        :param arr: (B, L, C)
        :return:
        '''
        B, L, C = arr.shape
        T = (B - 1) * stride + L
        out = np.zeros((T, L))
        x = arr.mean(axis=-1)  # (B, L, C) => (B, L)
        for i in range(L):
            out[i * stride: i * stride + B, i] = x[:, i]
        out = np.true_divide(out.sum(axis=1), (out != 0).sum(axis=1))  # get mean except 0
        return out

    def get_threshold(self, gt, anomaly_scores):
        '''
        Find the threshold according to Youden's J statistic,
        which maximizes (tpr-fpr)
        '''
        self.logger.info("Oracle Thresholding")
        fpr, tpr, thresholds = roc_curve(gt, anomaly_scores)
        J = tpr - fpr
        idx = np.argmax(J)
        best_threshold = thresholds[idx]
        self.logger.info(f"Best threshold found at: {best_threshold}, with fpr: {fpr[idx]}, tpr: {tpr[idx]}")
        return best_threshold

    def checkpoint(self, filepath):
        self.logger.info(f"checkpointing: {filepath} @Trainer - torch.save")
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.logger.info(f"loading: {filepath} @Trainer - torch.load_state_dict")
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.args.device)