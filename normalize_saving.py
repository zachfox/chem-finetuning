from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from torch.utils.data import Dataset
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
from sklearn.metrics import mean_squared_error
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
all_columns = 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed,fub_spec,fub_spec_transformed,log_clearance'
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
parser.add_argument('--tokenizer_name', type=str, default='./tokenizers/bert_tok', help='tokenizer from hugging face')
parser.add_argument('--tokenizer_type', type=str, default='bert', help='type for pretokenizer')
parser.add_argument('--model_name', type=str, default='./models', help='model from hugging face')
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
device = None
if args.device is None:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    device = torch.device('cuda:%d' % (rank % 6))
    run_id = '%s_%d' % (args.run_id, rank)
else:
    device = torch.device(args.device)
    run_id = args.run_id

log_file_name = '%s/training%s.log' % (args.output_directory, run_id)
log_file = open(log_file_name, 'w')

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
