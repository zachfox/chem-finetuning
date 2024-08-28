import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import time
import argparse
import rdkit
import rdkit.Chem
import csv
import json

import model
import train_utils as tu
import data_utils as du
import parser

import wandb

#########################################################################
# Arguments
#########################################################################
args = parser.parse_arguments()

# setup for mpi or specified device
device = torch.device("cuda")
run_id = args.run_id

logger = wandb.init(project="az finetuning", config=args)
wandb_flag = True

#########################################################################
# Training and Evaluation
#########################################################################
loss_fcn = None
if args.loss_type == "MSE":
    if args.ntasks == 1:
        loss_fcn = torch.nn.MSELoss()
    else:
        loss_fcn = tu.MaskMSELoss()
        # def loss_fcn(output, labels):
        #     mse_loss = torch.nn.functional.mse_loss(output, labels, reduction='mean')
        #     num_tasks = output.shape[1]
        #     return mse_loss / num_tasks

elif args.loss_type == "BCEWithLogitsLoss":
    loss_fcn = torch.nn.BCEWithLogitsLoss()

#########################################################################
# Read Train/Test/Val Data
#########################################################################
print("Preparing Data for Training...")
if args.tdc:
    train_x, train_y, val_x, val_y, test_x, test_y = du.load_data_admet(args)
else:
    train_x, train_y, val_x, val_y, test_x, test_y = du.load_data(args)

print(train_y)
train_data = du.RegressionDataset(
    train_x, train_y, args.tokenizer_name, args.tokenizer_type
)
val_data = du.RegressionDataset(val_x, val_y, args.tokenizer_name, args.tokenizer_type)
test_data = du.RegressionDataset(
    test_x, test_y, args.tokenizer_name, args.tokenizer_type
)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

#########################################################################
# Setup Simple Model
#########################################################################

simpleModel = model.Net(args, args.ntasks)
simpleModel = simpleModel.to(device)
optimizer = torch.optim.AdamW(
    simpleModel.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

#########################################################################
# Train Simple Model
#########################################################################

optimalVal = None
if args.validation_type == "R2":
    # for R2 want maximum value, for others want minimum
    optimalVal = np.NINF
else:
    optimalVal = np.Inf
optimalEpoch = -1
val_model_location = (
    args.output_directory
    + "/tuned_model"
    + run_id
    + "_val_"
    + args.validation_type
    + ".tar"
)

pytorch_total_params = sum(
    p.numel() for p in simpleModel.parameters() if p.requires_grad
)
print("Total Parameters:", pytorch_total_params)

for i in range(args.epochs):
#    print("Epoch:", i)

    trainTime, trainLoss = tu.train_epoch(
        simpleModel, loss_fcn, optimizer, train_loader, device
    )

    valTime, valLoss, valMSE, valMAE, valR2, valSR = tu.evaluate_model(
        simpleModel, loss_fcn, val_loader, device, args, split="val", logger=logger
    )

    testTime, testLoss, testMSE, testMAE, testR2, testSR = tu.evaluate_model(
        simpleModel, loss_fcn, test_loader, device, args, split="test", logger=logger
    )

    logger.log(
        {
            "epoch": i,
            "train time": trainTime,
            "train loss": trainLoss,
            "val mae": valMAE,
            "val time": valTime,
            "val loss": valLoss,
            "val rmse": np.sqrt(valMSE),
            "val R2": valR2,
            "val spearman": valSR,
            "test mae": testMAE,
            "test time": testTime,
            "test loss": testLoss,
            "test rmse": np.sqrt(testMSE),
            "test R2": testR2,
            "test spearman": testSR,
        }
    )

    # check on optimal value for validation set
    if args.validation_type == "R2":
        if valR2 > optimalVal:
            optimalVal = valR2
            optimalEpoch = i
            torch.save(
                {
                    "model_state_dict": simpleModel.state_dict(),
                },
                val_model_location,
            )
    elif args.validation_type == "MSE":
        if valMSE < optimalVal:
            optimalVal = valMSE
            optimalEpoch = i
            # print("New Optimal MSE")
            torch.save(
                {
                    "model_state_dict": simpleModel.state_dict(),
                },
                val_model_location,
            )
    else:
        if valLoss < optimalVal:
            optimalVal = valLoss
            optimalEpoch = i
            # print("New Optimal Loss")
            torch.save(
                {
                    "model_state_dict": simpleModel.state_dict(),
                },
                val_model_location,
            )

#########################################################################
# Load Val Minimum and Evaluate on Test Set
#########################################################################

print("Loading Val Optimal Model from Epoch:", optimalEpoch)

checkpoint = torch.load(val_model_location)
simpleModel.load_state_dict(checkpoint["model_state_dict"])

print("Model Evaluation Stats...")
if args.normalize:
    normalize = (mean_y, std_y)
else:
    normalize = None

(
    trainTime,
    trainLoss,
    trainMSE,
    trainMAE,
    trainR2,
    trainY,
    trainYPred,
) = tu.evaluate_model(
    simpleModel,
    loss_fcn,
    train_loader,
    device,
    args,
    write_predictions="train",
    header=args.score_columns,
    normalize=normalize,
    log_transform=args.log_transform,
    return_predictions=True,
    split="train",
)

print("Train Time:", trainTime)
print("Train Loss:", trainLoss)
print("Train RMSE:", np.sqrt(trainMSE))
print("Train R2:", trainR2)

valTime, valLoss, valMSE, valMAE, valR2, valSR, valY, valYPred = evaluate_model(
    simpleModel,
    loss_fcn,
    val_loader,
    device,
    args,
    write_predictions="val",
    header=args.score_columns,
    normalize=normalize,
    log_transform=args.log_transform,
    return_predictions=True,
    split="val",
)

print("Val Time:", valTime)
print("Val Loss:", valLoss)
print("Val RMSE:", np.sqrt(valMSE))
print("Val R2:", valR2)

(
    testTime,
    testLoss,
    testMSE,
    testMAE,
    testR2,
    testSR,
    testY,
    testYPred,
) = tu.evaluate_model(
    simpleModel,
    loss_fcn,
    test_loader,
    device,
    args,
    write_predictions="test",
    header=args.score_columns,
    normalize=normalize,
    log_transform=args.log_transform,
    return_predictions=True,
    split="test",
)

print("Test Time:", testTime)
print("Test Loss:", testLoss)
print("Test RMSE:", np.sqrt(testMSE))
print("Test R2:", testR2)

if wandb_flag:
    logger.log(
        {
            "epoch": i,
            "train time": trainTime,
            "train loss": trainLoss,
            "val mae": valMAE,
            "val time": valTime,
            "val loss": valLoss,
            "val rmse": np.sqrt(valMSE),
            "val R2": valR2,
            "val spearman": valSR,
            "test mae": testMAE,
            "test time": testTime,
            "test loss": testLoss,
            "test rmse": np.sqrt(testMSE),
            "test R2": testR2,
            "test spearman": testSR,
        }
    )
#    data = [[x, y] for (x, y) in zip(trainY, trainYPred)]
#    tableTrain = wandb.Table(data=data, columns=["class_x", "class_y"])
#    wandb.log({"train_parity": wandb.plot.scatter(table, "class_x", "class_y")})
#
#    data = [[x, y] for (x, y) in zip(testY, testYPred)]
#    tableTest = wandb.Table(data=data, columns=["class_x", "class_y"])
#    wandb.log({"test_parity": wandb.plot.scatter(table, "class_x", "class_y")})
#
#    data = [[x, y] for (x, y) in zip(valY, valYPred)]
#    tableVal = wandb.Table(data=data, columns=["class_x", "class_y"])
#    wandb.log({"val_parity": wandb.plot.scatter(table, "class_x", "class_y")})

