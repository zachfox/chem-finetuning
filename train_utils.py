import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

from tqdm import tqdm
import wandb

wandb_flag = True


class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()

    def forward(self, output, target):
        # Create a mask to identify NaN values in the target tensor
        mask = ~torch.isnan(target)
        # Apply the mask to both input and target tensors
        # output_masked = output[mask]
        # target_masked = target
        # Compute the mean squared error only on non-NaN values
        # loss = F.mse_loss(output_masked, target_masked)
        loss = F.l1_loss(output, target, reduction='none')*mask
        return loss.sum() / mask.sum()


def train_epoch(model, loss_fcn, optimizer, dataloader, device):
#    print("training epoch...")
    t = time.time()
    total_loss = 0.0
    model.train()
    batch_counter = 0

    for b, batch in enumerate(tqdm(dataloader, leave=False)):
        batch_ids = batch["ids"].to(device)
        batch_mask = batch["mask"].to(device)
        batch_labels = batch["labels"].to(device)
        output = model(batch_ids, batch_mask)
        loss_train = loss_fcn(output, batch_labels)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        batch_counter += 1

#    print("...done")
    return (time.time() - t), total_loss / batch_counter


def evaluate_model(
    model,
    loss_fcn,
    dataloader,
    device,
    args,
    return_predictions=False,
    split="train",
    logger=None
):
    t = time.time()
    total_loss = 0.0
    model.eval()

    all_outputs = []
    all_labels = []
    batch_counter = 0

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch_ids = batch["ids"].to(device)
            batch_mask = batch["mask"].to(device)
            batch_labels = batch["labels"].to(device)
            output = model(batch_ids, batch_mask)

            all_outputs.append(output)
            all_labels.append(batch_labels)
        
            loss_train = loss_fcn(output, batch_labels)
            total_loss += loss_train.item()
            batch_counter += 1

    all_outputs = torch.cat(all_outputs, dim=0).to('cpu').detach().numpy()
    all_labels = torch.cat(all_labels, dim=0).to('cpu').detach().numpy()
   
    if args.normalize:
        mean_y = np.loadtxt(args.output_directory + "/normalization_mean.txt")
        std_y = np.loadtxt(args.output_directory + "/normalization_std.txt")
        all_outputs = all_outputs*std_y + mean_y
   
    # ignore nan labels 
    islabel = ~np.isnan(all_labels.flatten())
    mse = mean_squared_error(all_labels.flatten()[islabel],
                             all_outputs.flatten()[islabel])
    mae = mean_absolute_error(all_labels.flatten()[islabel],
                              all_outputs.flatten()[islabel])
    r2 = r2_score(all_labels.flatten()[islabel],
                  all_outputs.flatten()[islabel])
    spearman = spearmanr(all_labels.flatten()[islabel],
                         all_outputs.flatten()[islabel])

    # make a prediction dataframe
    cols_true = [ col + '_true' for col in args.score_columns]
    cols_pred = [col + '_pred' for col in args.score_columns]
    pred_df = pd.DataFrame(all_outputs, columns=cols_pred)
    true_df = pd.DataFrame(all_labels, columns=cols_true)
    df = pd.concat([true_df, pred_df], axis=1)
    mae_df, r2_df, mse_df, spearman_df = compute_metrics(df)
    if wandb_flag:
        pred_table = wandb.Table(dataframe=df)
        mae_table = wandb.Table(dataframe=mae_df)
        r2_table = wandb.Table(dataframe=r2_df)
        mse_table = wandb.Table(dataframe=mse_df)
        spearman_table = wandb.Table(dataframe=spearman_df)
        # wandb.log({f"{split}_parity": wandb.plot.scatter(table, "true", "predicted")})
        wandb.log({f'{split} prediction table':pred_table, f'{split} mae table': mae_table, 
                    f'{split} r2 table':r2_table, f'{split} mse table': mse_table, f'{split} spearman table': spearman_table})
    if return_predictions:
        return (
            (time.time() - t),
            total_loss / batch_counter,
            mse,
            mae,
            r2,
            spearman,
            true_y_flat,
            pred_y_flat,
        )

    return (time.time() - t), total_loss / batch_counter, mse, mae, r2, spearman


def compute_metrics(df):
    # Initialize lists to hold the metrics data
    mae_data = []
    r2_data = []
    mse_data = []
    spearman_data = []

    pred_columns = [col for col in df.columns if col.endswith('_pred')]
    true_columns = [col for col in df.columns if col.endswith('_true')]

    for pred_col in pred_columns:
        true_col = pred_col.replace('_pred', '_true')
        if true_col in true_columns:
            y_pred = df[pred_col].dropna()
            y_true = df[true_col].dropna()
            common_index = y_pred.index.intersection(y_true.index)
            y_pred = y_pred.loc[common_index]
            y_true = y_true.loc[common_index]

            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)

            mae_data.append({'Prediction': pred_col, 'True': true_col, 'MAE': mae})
            r2_data.append({'Prediction': pred_col, 'True': true_col, 'R2': r2})
            mse_data.append({'Prediction': pred_col, 'True': true_col, 'MSE': mse})
            spearman_data.append({'Prediction': pred_col, 'True': true_col, 'Spearman Rank Correlation': spearman_corr})

    # Convert lists to DataFrames
    mae_df = pd.DataFrame(mae_data)
    r2_df = pd.DataFrame(r2_data)
    mse_df = pd.DataFrame(mse_data)
    spearman_df = pd.DataFrame(spearman_data)

    return mae_df, r2_df, mse_df, spearman_df 
