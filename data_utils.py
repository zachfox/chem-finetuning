import torch
import pandas as pd
import numpy as np
import json

from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import (
    BertPreTokenizer,
    Digits,
    Sequence,
    WhitespaceSplit,
)
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from torch.utils.data import Dataset, DataLoader, Subset

from tdc.benchmark_group import admet_group

import wandb

wandb_flag = False


class RegressionDataset(Dataset):
    def __init__(self, data_x, data_y, tokenizer_name, tokenizer_type, normalize=False):
        pretokenizer_dict = {
            "regex": Sequence(
                [
                    WhitespaceSplit(),
                    Split(
                        Regex(
                            r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
                        ),
                        behavior="isolated",
                    ),
                ]
            ),
            "bert": BertPreTokenizer(),
            "bert_digits": Sequence(
                [BertPreTokenizer(), Digits(individual_digits=True)]
            ),
            "digits": Sequence([WhitespaceSplit(), Digits(individual_digits=True)]),
        }
        # initialize tokenizer
        with open(tokenizer_name + "/config.json", "r") as f:
            tokenizer_config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_config
        )
        self.tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[
            tokenizer_type
        ]

        # convert raw data to tokens
        self.data_x = data_x
        self.data_y = data_y
        self.encodings = self.tokenizer(self.data_x, padding=True, return_tensors="pt")
        self.ids, self.mask = self._generateModelInputs(self.encodings)

    def _generateModelInputs(self, data):
        # check that appropriate info is present
        if "input_ids" not in data:
            return None
        if "attention_mask" not in data:
            return None
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        return input_ids, attention_mask

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        sample = {
            "labels": torch.tensor(self.data_y[idx], dtype=torch.float),
            "ids": self.ids[idx],
            "mask": self.mask[idx],
            "smiles": self.data_x[idx],
        }
        return sample


def randomize_smiles(input_smiles, requested_smiles, retries=2):
    output_smiles = set()
    retry_counter = 0
    while (len(output_smiles) < requested_smiles) and (retry_counter < retries):
        try:
            mol = rdkit.Chem.MolFromSmiles(input_smiles)
            atom_order = list(range(mol.GetNumAtoms()))
            np.random.shuffle(atom_order)
            new_mol = rdkit.Chem.RenumberAtoms(mol, atom_order)
            random_smiles = rdkit.Chem.MolToSmiles(new_mol, canonical=False)
            if (random_smiles != input_smiles) and (random_smiles not in output_smiles):
                output_smiles.add(random_smiles)
            else:
                retry_counter += 1
        except:
            retry_counter += 1

    return output_smiles


# def check_smiles(smiles):
#    try:
#        smiles = rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(row_split[header[args.data_column]]))
#        labels = [float(row_split[header[column]] for column in args.score_columns)]
#        all_data_x.append(smiles)
#        all_data_y.append(label)
#    except:
#        continue


def check_smiles_len(smiles, max_len):
    for k, smile in enumerate(smiles):
        if len(smile) > max_len:
            smiles[k] = smile[:max_len]
    return smiles


def augment_smiles(number_randomized_smiles):
    if number_randomized_smiles > 0:
        extra_x = []
        extra_y = []
        for i in range(len(train_x)):
            r_smiles = randomize_smiles(train_x[i], number_randomized_smiles)
            if r_smiles is not None:
                extra_x += list(r_smiles)
                extra_y += [train_y[i] for _ in range(len(r_smiles))]

        train_x += extra_x
        train_y += extra_y


def add_random_split(args, data):
    ntrain = int(len(data) * args.train_fraction)
    nval = int((len(data) - ntrain) / 2)
    ntest = len(data) - ntrain - nval
    split_column = ["train"] * ntrain + ["val"] * nval + ["test"] * ntest
    np.random.shuffle(split_column)
    data["split"] = split_column
    return data


def load_data(args):
    data = pd.read_csv(args.input_file_csv)
    data[args.data_column] = check_smiles_len(data[args.data_column], args.max_len)

    if args.split_column == "None":
        args.split_column = "split"
        data = add_random_split(args, data)

    train_data = data[data[args.split_column] == "train"]
    val_data = data[data[args.split_column] == "val"]
    test_data = data[data[args.split_column] == "test"]

    train_x = train_data[args.data_column].to_list()
    val_x = val_data[args.data_column].to_list()
    test_x = test_data[args.data_column].to_list()

    train_y = train_data[args.score_columns].values.tolist()
    val_y = val_data[args.score_columns].values.tolist()
    test_y = test_data[args.score_columns].values.tolist()

    if args.log_transform:
        train_y = np.log10(train_y)
        val_y = np.log10(val_y)
        test_y = np.log10(test_y)

    if args.normalize:
        train_y = np.array(train_y)
        mean_y = np.mean(train_y, axis=0)
        std_y = np.std(train_y, axis=0)
        for i, std in enumerate(std_y):
            if std < 1e-5:
                std_y[i] = 1.0
        train_y = (train_y-mean_y)/std_y
        # save the normalization to the ouput directory
        np.savetxt(args.output_directory + "/normalization_mean.txt", mean_y)
        np.savetxt(args.output_directory + "/normalization_std.txt", std_y)

    if args.max_scale > 0.0:
        train_y = [np.clip(y / args.max_scale, 0.0, 1.0) for y in train_y]
        val_y = [np.clip(y / args.max_scale, 0.0, 1.0) for y in train_y]

    if wandb_flag:
        table_train = wandb.Table(data=train_y.tolist(), columns=["train y"])
        wandb.log(
            {
                "train_label_histogram": wandb.plot.histogram(
                    table_train, "train y", title="Train labels"
                )
            }
        )

        table_valtest = wandb.Table(data=val_y, columns=["val y"])
        wandb.log(
            {
                "val_label_histogram": wandb.plot.histogram(
                    table_valtest, "val y", title="val labels"
                )
            }
        )

    print("Total Samples Read: %d" % len(data))

    print("Training Samples: %d" % len(train_x))
    print("Validation Samples: %d" % len(val_x))
    print("Testing Samples: %d" % len(test_x))
    return train_x, train_y, val_x, val_y, test_x, test_y


def load_data_admet(args):
    group = admet_group(path="data/")
    # benchmark = group.get("lipophilicity_astrazeneca")
    benchmark = group.get("Clearance_Hepatocyte_AZ")
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]

    train_x = train_val[args.data_column].to_list()
    val_x = test[args.data_column].to_list()
    test_x = test[args.data_column].to_list()

    train_y = train_val[["Y"]].values.tolist()
    val_y = test[["Y"]].values.tolist()
    test_y = test[["Y"]].values.tolist()

    if wandb_flag:
        table_train = wandb.Table(data=[[train_y]], columns=["train y"])
        wandb.log(
            {
                "train_label_histogram": wandb.plot.histogram(
                    table_train, "train y", title="Train labels"
                )
            }
        )

        table_valtest = wandb.Table(data=[[val_y]], columns=["val y"])
        wandb.log(
            {
                "val_label_histogram": wandb.plot.histogram(
                    table_valtest, "val y", title="val labels"
                )
            }
        )

    if args.log_transform:
        train_y = np.log10(train_y)
        val_y = np.log10(val_y)
        test_y = np.log10(test_y)

    if args.normalize:
        train_y = np.atleast_2d(np.array(train_y))
        mean_y = np.mean(train_y, axis=0)
        std_y = np.std(train_y, axis=0)
        for i, std in enumerate(std_y):
            if std < 1e-5:
                std_y[i] = 1.0
        train_y = (train_y-mean_y)/std_y
        # save the normalization to the ouput directory
        np.savetxt(args.output_directory + "/normalization_mean.txt", mean_y)
        np.savetxt(args.output_directory + "/normalization_std.txt", std_y)

    if args.max_scale > 0.0:
        train_y = [np.clip(y / args.max_scale, 0.0, 1.0) for y in train_y]
        val_y = [np.clip(y / args.max_scale, 0.0, 1.0) for y in train_y]

    print("Training Samples: %d" % len(train_x))
    print("Validation Samples: %d" % len(val_x))
    print("Testing Samples: %d" % len(test_x), flush=True)
    return train_x, train_y, val_x, val_y, test_x, test_y
