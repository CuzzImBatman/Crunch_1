import os
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from dataset import CLUSTER_BRAIN
from pathlib import Path
# from lr_scheduler import LR_Scheduler
from torch.utils.data import Sampler
from collections import defaultdict
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from model import ImageEncoder
import pandas as pd
import torch.nn as nn

from sklearn.linear_model import Ridge  # Regression model
# torch.multiprocessing.set_start_method('spawn')
import pickle
class CustomBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.groups = defaultdict(list)

        # Group items by ID
        for idx, item in enumerate(dataset):
            self.groups[item['id']].append(idx)

    def __iter__(self):
        group_keys = list(self.groups.keys())
        
        # Shuffle group order if shuffle is enabled
        if self.shuffle:
            random.shuffle(group_keys)
        
        # For each group ID, shuffle items within the group if shuffle is enabled
        shuffled_indices = []
        for key in group_keys:
            indices = self.groups[key][:]
            if self.shuffle:
                random.shuffle(indices)  # Shuffle items within each group ID
            shuffled_indices.extend(indices)
        
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.dataset)

import warnings
from scipy.stats import pearsonr, ConstantInputWarning

def train_and_evaluate(model, train_loader, val_loader, args, max_iter=1000, alpha=None, optimizer=None, device='cuda:0'):
    """
    Combines training and evaluation for Ridge regression model in a memory-efficient manner.

    :param model: Ridge regression model.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param args: Training configuration arguments.
    :param max_iter: Number of training epochs.
    :param alpha: Regularization strength for Ridge regression.
    :param optimizer: Optimizer for training the model.
    :param device: Device to perform training and evaluation ('cuda' or 'cpu').
    """
    model = model.to(device)
    alpha = alpha or 100 / (model.linear.weight.shape[1] * model.linear.out_features)
    print(f"Training Ridge regression model with alpha={alpha}...")

    # Training loop
    for epoch in range(max_iter):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch['feature'].to(device, dtype=torch.float32)
            targets = batch['expression'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            predictions = model(features)
            loss = model.compute_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{max_iter}], Train Loss: {avg_train_loss:.4f}")

    # Validation
    print("Evaluating on validation set...")
    model.eval()
    preds_all, y_test_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            features = batch['feature'].to(device, dtype=torch.float32)
            targets = batch['expression'].to(device, dtype=torch.float32)
            predictions = model(features)

            preds_all.append(predictions.cpu().numpy())
            y_test_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    y_test_all = np.concatenate(y_test_all, axis=0)

    # Compute evaluation metrics
    results = compute_metrics(y_test_all, preds_all)
    print("Validation Metrics:", results)

    # Save results
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'regression_results.csv')
    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save model
    model_file = os.path.join(output_dir, 'ridge_model.pth')
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    return results

def compute_metrics(y_test, preds_all, genes=None):
    """
    Computes metrics L2 errors R2 scores and Pearson correlations for each target/gene.

    :param y_test: Ground truth values (numpy array of shape [n_samples, n_targets]).
    :param preds_all: Predictions for all targets (numpy array of shape [n_samples, n_targets]).
    :param genes: Optional list of gene names corresponding to targets.
    :return: A dictionary containing metrics.
    """

    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []

    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        # Compute L2 error
        l2_error = float(np.mean((preds - target_vals) ** 2))

        # Compute R2 score
        total_variance = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if total_variance == 0:
            r2_score = 0.0
        else:
            r2_score = float(1 - np.sum((target_vals - preds) ** 2) / total_variance)

        # Compute Pearson correlation
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConstantInputWarning)
            try:
                pearson_corr, _ = pearsonr(target_vals, preds)
                pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
            except ConstantInputWarning:
                pearson_corr = 0.0

        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)

        # Record gene-specific Pearson correlation
        if genes is not None:
            pearson_genes.append({
                'name': genes[i],
                'pearson_corr': pearson_corr
            })

    # Compile results
    results = {
        'pearson_mean': float(np.mean(pearson_corrs)),
        'l2_errors_mean': float(np.mean(errors)),
        'r2_scores_mean': float(np.mean(r2_scores)),
        # 'l2_errors': list(errors),
        # 'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_genes if genes is not None else list(pearson_corrs),
        'pearson_std': float(np.std(pearson_corrs)),
        'l2_error_q1': float(np.percentile(errors, 25)),
        'l2_error_q2': float(np.median(errors)),
        'l2_error_q3': float(np.percentile(errors, 75)),
        'r2_score_q1': float(np.percentile(r2_scores, 25)),
        'r2_score_q2': float(np.median(r2_scores)),
        'r2_score_q3': float(np.percentile(r2_scores, 75)),
    }

    return results

class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, predictions, targets):
        mse_loss = torch.mean((predictions - targets) ** 2)
        regularization = self.alpha * torch.norm(self.linear.weight, p=2) ** 2
        return mse_loss + regularization
def train_test_reg(model,X_train, X_test, y_train, y_test,
                   max_iter=1000, random_state=0, genes=None
                   , alpha=None
                   ,optimizer =None
                   , method='ridge', device='cuda:0'):
    """ Train a regression model and evaluate its performance on test data """

    if method == 'ridge':
        # Convert data to PyTorch tensors and move to GPU
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        alpha = alpha or 100 / (X_train.shape[1] * y_train.shape[1])
        print(f"Ridge: using alpha: {alpha}")

        # Initialize Ridge Regression model
        # model = RidgeRegression(input_dim, output_dim, alpha).to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        # Training loop
        for epoch in range(max_iter):
            model.train()
            batch_size=1024
            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                predictions = model(batch_X)
                optimizer.zero_grad()
                loss = model.compute_loss(predictions, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{max_iter}], Loss: {loss.item():.4f}")

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            preds_all = model(X_test).cpu().numpy()
            y_test = y_test.cpu().numpy()

        # Compute metrics
        results = compute_metrics(y_test, preds_all, genes=genes)
        print(results)
        return  results




def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4096, help='patch_size')

    parser.add_argument('--embed_dim', type=int, default=1024, help="The dimension of instance-level representations")
    parser.add_argument('--patch_size', type=int, default=112, help='patch_size')
    parser.add_argument('--utils', type=str, default=None, help='utils path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=460)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_dir', default='./', help='path where to save')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')
    parser.add_argument('--embed_dir', type=str, default='/content/preprocessed')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--local', type=bool, default=False)
    parser.add_argument('--encoder_mode', type=bool, default=False)

    return parser.parse_args()




def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Dataset and DataLoader preparation
    dir = args.embed_dir
    NAMES = ['DC1', 'DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
    if args.demo:
        NAMES = NAMES[:2]

    train_dataset = CLUSTER_BRAIN(emb_folder=dir, train=True, split=True, name_list=NAMES)
    print(f"Train dataset size: {len(train_dataset)}")

    val_dataset = [CLUSTER_BRAIN(emb_folder=dir, train=False, split=True, name_list=[name]) for name in NAMES]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )
    val_loaders = [
        DataLoader(
            v_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True
        )
        for v_set in val_dataset
    ]

    if args.local:
        print("Running locally")
        train_loaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
        val_loaders = [
            DataLoader(v_set, batch_size=args.batch_size, shuffle=False, pin_memory=False)
            for v_set in val_dataset
        ]

    # Ridge Regression Model Initialization
    input_dim = train_dataset[0]["feature"].shape[0]
    output_dim = train_dataset[0]["expression"].shape[1]
    alpha = 100 / (input_dim * output_dim)  # Default alpha value if not specified
    model = RidgeRegression(input_dim=input_dim, output_dim=output_dim, alpha=alpha).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Train the Ridge Regression model
    train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(val_dataset),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
        ),
        args=args,
        max_iter=args.epochs,
        alpha=alpha
        ,optimizer=optimizer
        # ,method="ridge"
    )

    
            

if __name__ == '__main__':
    opt = parse()
    # torch.multiprocessing.set_start_method('spawn')
    main(opt)
