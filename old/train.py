from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import Dataset
from torch.nn.functional import relu
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import WhitespaceSplit
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import time
import argparse
import rdkit
import rdkit.Chem
import csv
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score



# from mpi4py import MPI

#########################################################################
# Pretokenizer options
#########################################################################

pretokenizer_dict = {
    'regex': Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')]),
    'bert': BertPreTokenizer(),
    'bert_digits': Sequence([BertPreTokenizer(), Digits(individual_digits=True)]),
    'digits': Sequence([WhitespaceSplit(), Digits(individual_digits=True)])
}

#########################################################################
# Arguments
#########################################################################

# all_columns = 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed'
# all_columns = 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed,fub_spec,fub_spec_transformed,log_clearance'
all_columns = 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed,fub_spec,fub_spec_transformed,log_clearance,lipophilicity_az,clearance_az'

parser = argparse.ArgumentParser()

# input/output
parser.add_argument('input_file_csv', type=str, help='location of input data csv file')
parser.add_argument('--data_column', type=str, default='SMILES', help='data column in csv file')
parser.add_argument('--score_columns', type=str, default=all_columns, help='comma separated score columns in csv file')
parser.add_argument('--output_directory', type=str, default='./results/', help='output directory for model, predictions, and stats')
parser.add_argument('--run_id', type=str, default='_run', help='run id for model')

# data options
parser.add_argument('--train_fraction', type=float, default=0.8, help='fraction of samples for training')
parser.add_argument('--max_scale', type=float, default=-1.0, help='maximum used to divide scores')
parser.add_argument('--number_randomized_smiles', type=int, default=0, help='option to add randomized smiles to training')
parser.add_argument('--normalize', action='store_true', default=False, help='option to normalize data to mean 0 and standard deviation 1')
parser.add_argument('--log_transform', action='store_true', default=False, help='option to apply log to training values')

# pre-trained model information
parser.add_argument('--tokenizer_name', type=str, default='/lustre/or-scratch/cades-ccsd/z6f/lombardo-ft/tokenizers/bert_tok', help='tokenizer from hugging face')
parser.add_argument('--tokenizer_type', type=str, default='bert', help='type for pretokenizer')
parser.add_argument('--model_name', type=str, default='/lustre/or-scratch/cades-ccsd/z6f/lombardo-ft/models/', help='model from hugging face')
parser.add_argument('--use_random_weights', action='store_true', default=False, help='option to use random weights for pre-trained model')

# training options
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--dropout_fraction', type=float, default=0.0, help='dropout fraction on fully-connected layer')
parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'BCEWithLogitsLoss'], help='type of loss function for regression')
parser.add_argument('--validation_type', type=str, default='MSE', choices=['MSE', 'R2', 'Loss'], help='Metric to select best validation model')
parser.add_argument('--device', type=str, default=None, help='device for training')
parser.add_argument('--max_len', type=float, default=512, help='maximum sequence length')

args = parser.parse_args()

# split score columns
args.score_columns = args.score_columns.split(',')
args.ntasks = len(args.score_columns)

# setup for mpi or specified device
device = torch.device('cuda')
# if args.device is None:
#     from mpi4py import MPI
#     rank = MPI.COMM_WORLD.Get_rank()
#     device = torch.device('cuda:%d' % (rank % 6))
#     run_id = '%s_%d' % (args.run_id, rank)
# else:
#     device = torch.device(args.device)
run_id = args.run_id

log_file_name = '%s/training%s.log' % (args.output_directory, run_id)
log_file = open(log_file_name, 'w')

import wandb
logger = wandb.init(project="az finetuning", config=args)
wandb_flag = True

#try:
#
#except: 
#    wandb_flag = False
#    print("no wandb")

# print(args, file=log_file)
print(args)

#########################################################################
# Dataset
#########################################################################

class RegressionDataset(Dataset):
    def __init__(self, data_x, data_y, tokenizer_name, normalize=False):
        
        # initialize tokenizer
        with open(tokenizer_name + '/config.json', 'r') as f:
            tokenizer_config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_config)
        self.tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[args.tokenizer_type]

        # convert raw data to tokens
        self.data_x = data_x
        self.data_y = data_y        
        self.encodings = self.tokenizer(self.data_x, padding=True, return_tensors='pt')
        self.ids, self.mask = self._generateModelInputs(self.encodings)

    def _generateModelInputs(self, data):

        # check that appropriate info is present
        if 'input_ids' not in data:
            return None
        if 'attention_mask' not in data:
            return None
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        return input_ids, attention_mask
    
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
    
        sample = {'labels': torch.tensor(self.data_y[idx],dtype=torch.float),
                  'ids': self.ids[idx],
                  'mask': self.mask[idx],
                  'smiles': self.data_x[idx]}
        return sample    

#########################################################################
# Initial Dataset Processing
#########################################################################

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

is_header = True
header = {}
all_data_x = []
all_data_y = []
with open(args.input_file_csv, 'r') as f:
    reader = csv.reader(f)
    for row_split in reader:
        if is_header:
            counter = 0
            for elem in row_split:
                if len(elem) > 0:
                    header[elem] = counter
                counter += 1
            is_header = False
            continue
        smiles = rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(row_split[header[args.data_column]]))
        if len(smiles) > args.max_len:
            smiles = smiles[:args.max_len]
        labels = []

        # for column in args.score_columns:
        #     if column in args.log_columns:
        # if args.log_columns is not None:
        
        if not args.log_transform:
            labels = [float(row_split[header[column]]) for column in args.score_columns]
        else:
            labels = [np.log10(float(row_split[header[column]])) for column in args.score_columns]
                
        all_data_x.append(smiles)
        all_data_y.append(labels)


#        try:
#            smiles = rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(row_split[header[args.data_column]]))
#            labels = [float(row_split[header[column]] for column in args.score_columns)]
#            all_data_x.append(smiles)
#            all_data_y.append(label)
#        except:
#            continue

if args.normalize:
    if args.ntasks == 1:
        mean_y = np.mean(all_data_y)  
        std_y = np.std(all_data_y)
        if std_y < 1e-5:
            std_y = 1.0
        print('Normalizing Data %0.6f, %0.6f' % (mean_y, std_y), file=log_file, flush=True)
        all_data_y = [(y - mean_y) / std_y for y in all_data_y]
    elif args.ntasks > 1:
        mean_y = np.mean(np.array(all_data_y),axis=0)  
        std_y = np.std(np.array(all_data_y),axis=0)
        for i,std in enumerate(std_y):
            if std < 1e-5:
                std_y[i] = 1.0
        # print('Normalizing Data %0.6f, %0.6f' % (mean_y, std_y), file=log_file, flush=True)
        all_data_y = [[(task - mean_y[i]) / std_y[i] for i,task in enumerate(y) ] for y in all_data_y ]
        # save the normalization to the ouput directory
        np.savetxt(args.output_directory + '/normalization_mean.txt', mean_y)
        np.savetxt(args.output_directory + '/normalization_std.txt', std_y)


if args.max_scale > 0.0:
    all_data_y = [np.clip(y/args.max_scale, 0.0, 1.0) for y in all_data_y]

print('Total Samples Read: %d' % len(all_data_x), file=log_file, flush=True)
training_samples = int(len(all_data_x)*args.train_fraction)
validation_stop = int(len(all_data_x)*(args.train_fraction + 1.0) / 2.0)
all_indices = np.random.permutation(len(all_data_x))

train_x = [all_data_x[i] for i in all_indices[:training_samples]]
val_x = [all_data_x[i] for i in all_indices[training_samples:validation_stop]]
test_x = [all_data_x[i] for i in all_indices[validation_stop:]]

train_y = [all_data_y[i] for i in all_indices[:training_samples]]
val_y = [all_data_y[i] for i in all_indices[training_samples:validation_stop]]
test_y = [all_data_y[i] for i in all_indices[validation_stop:]]

if args.number_randomized_smiles > 0:
    extra_x = []
    extra_y = []
    for i in range(len(train_x)):
        r_smiles = randomize_smiles(train_x[i], args.number_randomized_smiles)
        if r_smiles is not None:
            extra_x += list(r_smiles)
            extra_y += [train_y[i] for _ in range(len(r_smiles))]
    
    train_x += extra_x
    train_y += extra_y

print('Training Samples: %d' % len(train_x), file=log_file)
print('Validation Samples: %d' % len(val_x), file=log_file)
print('Testing Samples: %d' % len(test_x), file=log_file, flush=True)

#########################################################################
# Training and Evaluation
#########################################################################

loss_fcn = None
if args.loss_type == 'MSE':
    if args.ntasks == 1: 
        loss_fcn = torch.nn.MSELoss()
    else:
        def loss_fcn(output, labels):
            mse_loss = torch.nn.functional.mse_loss(output, labels, reduction='mean')
            num_tasks = output.shape[1]
            return mse_loss / num_tasks

elif args.loss_type == 'BCEWithLogitsLoss':
    loss_fcn = torch.nn.BCEWithLogitsLoss()

def trainEpoch(model, optimizer, dataloader):

    t = time.time()
    total_loss = 0.0
    model.train()    
    batch_counter = 0

    for b,batch in enumerate(dataloader):
        
        batch_ids = batch['ids'].to(device)
        batch_mask = batch['mask'].to(device)
        batch_labels = batch['labels'].to(device)
        output = model(batch_ids, batch_mask)
        if len(output.shape)==1: 
            loss_train = loss_fcn(output.unsqueeze(1), batch_labels)
        else:
            loss_train = loss_fcn(output, batch_labels)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        batch_counter += 1

    return (time.time() - t), total_loss / batch_counter

def evaluateModel(model, dataloader, write_predictions='', header=None, normalize=None, log_transform=False, return_predictions=False, split='train'):

    t = time.time()
    total_loss = 0.0
    model.eval()
    batch_counter = 0

    true_y = []
    pred_y = []
    smiles = []

    with torch.no_grad():
        for _,batch in enumerate(dataloader):
            
            batch_ids = batch['ids'].to(device)
            batch_mask = batch['mask'].to(device)
            batch_labels = batch['labels'].to(device)
            output = model(batch_ids, batch_mask)
            smiles += [x for x in batch['smiles']]

            loss_train = loss_fcn(output, batch_labels)
            total_loss += loss_train.item()
            batch_counter += 1

            # save labels and preds for metric
            true_y += [[k.item() for k in x] for x in batch_labels]
            if args.loss_type == 'BCEWithLogitsLoss':
                pred_y += [k.item() for x in torch.nn.Sigmoid()(output) for k in x]
            else:
                if len(output.shape) == 1:
                    pred_y += [[k.item() for k in x] for x in output.unsqueeze(1)]
                else:
                    pred_y += [[k.item() for k in x] for x in output]
    true_y_flat  = [k for x in true_y for k in x]
    pred_y_flat  = [k for x in pred_y for k in x]
    mse = mean_squared_error(true_y_flat, pred_y_flat)
    mae = mean_absolute_error(true_y_flat, pred_y_flat)
    r2 = r2_score(true_y_flat, pred_y_flat)

    if wandb_flag:
        data = [[x, y] for (x, y) in zip(true_y_flat, pred_y_flat)]
        table = wandb.Table(data=data, columns=["true", "predicted"])
        wandb.log({f"{split}_parity": wandb.plot.scatter(table, "class_x", "class_y")})


#    if len(write_predictions) > 0:
#        with open(args.output_directory + '/predictions_' + write_predictions + run_id + '.csv', 'w') as f:
#            if header is not None:
#                f.write('smiles,'+','.join([col+'_true' for col in header])+','+','.join([col+'_pred' for col in header])+'\n')
#            for i in range(len(pred_y)):
#                if normalize is not None: # undo the scaling
#                    true_y_string = ','.join([str(true*normalize[1][k]+normalize[0][k]) for k,true in enumerate(true_y[i])])
#                    pred_y_string = ','.join([str(pred*normalize[1][k]+normalize[0][k]) for k,pred in enumerate(pred_y[i])])
#                elif log_transform:
#                    true_y_string = ','.join([str(10**true) for true in true_y[i]])
#                    pred_y_string = ','.join([str(10**pred) for pred in pred_y[i]])
#                else:
#                    true_y_string = ','.join([str(true) for true in true_y[i]])
#                    pred_y_string = ','.join([str(pred) for pred in pred_y[i]])
#                line = smiles[i] + ',' + true_y_string + ',' + pred_y_string + '\n'
#                f.write(line)

    if return_predictions:
        return (time.time() - t), total_loss / batch_counter, mse, mae, r2, true_y_flat, pred_y_flat
        
    return (time.time() - t), total_loss / batch_counter, mse, mae, r2 

#########################################################################
# Simple Model to Use Embeddings
#########################################################################

class Net(torch.nn.Module):
    def __init__(self, output_dim):
        super(Net, self).__init__()
        hidden_dim = 256
        self.output_dim = output_dim
        self.embedding = None
        if args.use_random_weights:
            config = AutoConfig.from_pretrained(args.model_name)
            self.embedding = AutoModel.from_config(config)
        else:
            self.embedding = AutoModel.from_pretrained(args.model_name)
        self._embedding_dim = self.embedding.config.hidden_size
        self.dropout = torch.nn.Dropout(p=args.dropout_fraction)

        # self.fc0 = torch.nn.Linear(self._embedding_dim, output_dim)
        self.fc0 = torch.nn.Linear(self._embedding_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):

        self.fc0.weight.data.normal_(mean=0.0, std=0.02)
        self.fc0.bias.data.zero_()
        self.fc1.weight.data.normal_(mean=0.0, std=0.02)
        self.fc1.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[0][:,0]
        x = self.dropout(x)
        x = self.fc0(x)
        x = relu(x)
        if self.output_dim ==  1:
            x = self.fc1(x)
        else:
            x = self.fc1(x).squeeze(-1)
        return x

#########################################################################
# Read Train/Test/Val Data
#########################################################################

print('Preparing Data for Training...', file=log_file, flush=True)
train_data = RegressionDataset(train_x, train_y, args.tokenizer_name)
val_data = RegressionDataset(val_x, val_y, args.tokenizer_name)
test_data = RegressionDataset(test_x, test_y, args.tokenizer_name)

train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
val_loader = DataLoader(val_data,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_data,batch_size=args.batch_size,shuffle=False)

#########################################################################
# Setup Simple Model
#########################################################################

simpleModel = Net(args.ntasks)
simpleModel = simpleModel.to(device)

optimizer = torch.optim.AdamW(simpleModel.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#########################################################################
# Train Simple Model
#########################################################################

optimalVal = None
if args.validation_type == 'R2':
    # for R2 want maximum value, for others want minimum
    optimalVal = np.NINF
else:
    optimalVal = np.Inf
optimalEpoch = -1
val_model_location = args.output_directory + '/tuned_model' + run_id + '_val_' + args.validation_type + '.tar'

pytorch_total_params = sum(p.numel() for p in simpleModel.parameters() if p.requires_grad)
print('Total Parameters:', pytorch_total_params, file=log_file, flush=True)

for i in range(args.epochs):
    print('Epoch:', i, file=log_file)

    trainTime, trainLoss = trainEpoch(simpleModel, optimizer, train_loader)
    
    

    print('Train Time:', trainTime, file=log_file)
    print('Train Loss:', trainLoss, file=log_file)

    valTime, valLoss, valMSE, valMAE, valR2 = evaluateModel(simpleModel, val_loader, split='val')

    print('Val Time:', valTime, file=log_file)
    print('Val Loss:', valLoss, file=log_file)
    print('Val RMSE:', np.sqrt(valMSE), file=log_file)
    print('Val R2:', valR2, file=log_file)

    testTime, testLoss, testMSE, testMAE, testR2 = evaluateModel(simpleModel, test_loader, split='test')

    print('Test Time:', testTime, file=log_file)
    print('Test Loss:', testLoss, file=log_file)
    print('Test RMSE:', np.sqrt(testMSE), file=log_file)
    print('Test R2:', testR2, file=log_file)

    logger.log({'epoch': i, 'train time': trainTime, 'train loss': trainLoss,
                'val mae':valMAE, 'val time': valTime, 'val loss': valLoss, 'val rmse': np.sqrt(valMSE),'val R2': valR2, 
                'test mae':testMAE, 'test time': testTime, 'test loss': testLoss, 'test rmse': np.sqrt(testMSE),'test R2': testR2}
              )
                     
    # check on optimal value for validation set
    if args.validation_type == 'R2':
        if valR2 > optimalVal:
            optimalVal = valR2
            optimalEpoch = i
            print('New Optimal R2', file=log_file, flush=True)
            torch.save({
                'model_state_dict': simpleModel.state_dict(),
            }, val_model_location)
    elif args.validation_type == 'MSE':
        if valMSE < optimalVal:
            optimalVal = valMSE
            optimalEpoch = i
            print('New Optimal MSE', file=log_file, flush=True)
            torch.save({
                'model_state_dict': simpleModel.state_dict(),
            }, val_model_location)
    else:
        if valLoss < optimalVal:
            optimalVal = valLoss
            optimalEpoch = i
            print('New Optimal Loss', file=log_file, flush=True)
            torch.save({
                'model_state_dict': simpleModel.state_dict(),
            }, val_model_location)        

#########################################################################
# Load Val Minimum and Evaluate on Test Set
#########################################################################

print('Loading Val Optimal Model from Epoch:', optimalEpoch, file=log_file, flush=True)

checkpoint = torch.load(val_model_location)
simpleModel.load_state_dict(checkpoint['model_state_dict'])

print('Model Evaluation Stats...', file=log_file, flush=True)
if args.normalize:
    normalize = (mean_y, std_y)
else:
    normalize = None

trainTime, trainLoss, trainMSE, trainMAE, trainR2, trainY, trainYPred = evaluateModel(simpleModel, train_loader, write_predictions='train', header=args.score_columns, normalize=normalize, log_transform=args.log_transform, return_predictions=True, split='train')

print('Train Time:', trainTime, file=log_file)
print('Train Loss:', trainLoss, file=log_file)
print('Train RMSE:', np.sqrt(trainMSE), file=log_file)
print('Train R2:', trainR2, file=log_file)

valTime, valLoss, valMSE, valMAE, valR2, valY, valYPred= evaluateModel(simpleModel, val_loader, write_predictions='val', header=args.score_columns, normalize=normalize, log_transform=args.log_transform, return_predictions=True, split='val')

print('Val Time:', valTime, file=log_file)
print('Val Loss:', valLoss, file=log_file)
print('Val RMSE:', np.sqrt(valMSE), file=log_file)
print('Val R2:', valR2, file=log_file)

testTime, testLoss, testMSE, testMAE, testR2, testY, testYPred = evaluateModel(simpleModel, test_loader, write_predictions='test', header=args.score_columns, normalize=normalize, log_transform=args.log_transform, return_predictions=True, split='test')

print('Test Time:', testTime, file=log_file)
print('Test Loss:', testLoss, file=log_file)
print('Test RMSE:', np.sqrt(testMSE), file=log_file)
print('Test R2:', testR2, file=log_file)

if wandb_flag:
    logger.log({'train time': trainTime, 'train loss': trainLoss,
                'val mae':valMAE, 'val time': valTime, 'val loss': valLoss, 'val rmse': np.sqrt(valMSE),'val R2': valR2, 
                'test mae':testMAE, 'test time': testTime, 'test loss': testLoss, 'test rmse': np.sqrt(testMSE),'test R2': testR2})
#if wandb_flag:
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

log_file.close()
