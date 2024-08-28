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


#########################################################################
# Dataset
#########################################################################
class InferenceDataset(Dataset):
    def __init__(self, data_x, tokenizer_name, normalize=False):

        pretokenizer_dict = {
            'regex': Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')]),
            'bert': BertPreTokenizer(),
            'bert_digits': Sequence([BertPreTokenizer(), Digits(individual_digits=True)]),
            'digits': Sequence([WhitespaceSplit(), Digits(individual_digits=True)])
        }
        
        # initialize tokenizer
        with open(tokenizer_name + '/config.json', 'r') as f:
            tokenizer_config = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_config)
        self.tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[args.tokenizer_type]

        # convert raw data to tokens
        self.data_x = data_x
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
    
        sample = {'ids': self.ids[idx],
                  'mask': self.mask[idx],
                  'smiles': self.data_x[idx]}
        return sample    

#########################################################################
# Simple Model to Use Embeddings
#########################################################################

class Net(torch.nn.Module):
    def __init__(self, output_dim):
        super(Net, self).__init__()
        self.output_dim = output_dim
        self.embedding = None
        if args.use_random_weights:
            config = AutoConfig.from_pretrained(args.model_name)
            self.embedding = AutoModel.from_config(config)
        else:
            self.embedding = AutoModel.from_pretrained(args.model_name)
        self._embedding_dim = self.embedding.config.hidden_size
        self.dropout = torch.nn.Dropout(p=args.dropout_fraction)

        self.fc0 = torch.nn.Linear(self._embedding_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):

        self.fc0.weight.data.normal_(mean=0.0, std=0.02)
        self.fc0.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[0][:,0]
        x = self.dropout(x)
        if self.output_dim ==  1:
            x = self.fc0(x)
        else:
            x = self.fc0(x).squeeze(-1)
        return x

#########################################################################
# Property generation
#########################################################################
def get_inference_loader(population, args):
    infer_x = population._sequences
    infer_data = InferenceDataset(infer_x, args.tokenizer_name)
    infer_loader = DataLoader(infer_data, batch_size=args.batch_size, shuffle=False) 
    return infer_loader

def load_model(args, path_to_model):
    simpleModel = Net(args.ntasks)
    simpleModel = simpleModel.to(args.device)
    simpleModel.load_state_dict(torch.load(path_to_model)['model_state_dict'])
    simpleModel.eval()
    return simpleModel

def get_properties(population,args):
    # get the property models
    args.ntasks = 7
    multitask_model = load_model( args, 'results/{0}/tuned_model_run_0_val_MSE.tar'.format(4078406))
    args.ntasks = 1
    IC50_model = load_model( args, 'results/{0}/tuned_model_run_0_val_MSE.tar'.format(4127260))
    clearance_model = load_model( args, 'results/{0}/tuned_model_run_0_val_MSE.tar'.format(4128162))

    multitask_map = {'Fu':5, 'MW': 0, 'Base':4,'Acid':3,'LogD':2} # output indices to features
    IC50_map = {'IC50':0}
    clearance_map = {'Clearance':0}

    models = [multitask_model, IC50_model, clearance_model]
    maps = [multitask_map, IC50_map, clearance_map ]

    # get the data loader
    infer_loader = get_inference_loader(population, args)
   
    property_dict = {} 
    for i,model in enumerate( models ):
        property_dict = infer_model(model, infer_loader, maps[i], property_dict, args)

    return property_dict

def infer_model(model, dataloader, properties, property_dict, args):
    with torch.no_grad():
        for _,batch in enumerate(dataloader):
            
            batch_ids = batch['ids'].to(args.device)
            batch_mask = batch['mask'].to(args.device)
            output = model(batch_ids, batch_mask)
            
            for k,smiles in enumerate(batch['smiles']):
                if smiles not in property_dict:
                    property_dict[smiles] = {}
                for property in properties:
                    property_dict[smiles][property] = output[k,properties[property]].item()
    return property_dict

if __name__ == '__main__':
    class Population:
        def __init__(self, smiles):
            self._sequences = smiles
    
    class Arguments:
        def __init__(self):   
            return 
    args = Arguments()
    args.tokenizer_name = './tokenizers/bert_tok'
    args.tokenizer_type = 'bert'
    args.use_random_weights = False
    args.model_name = './models'
    args.dropout_fraction = 0
    args.device = 'cuda'
    args.batch_size = 32

    population = Population(['O=C1CC[C@@]2(O)[C@H]3Cc4ccc(O)c5c4[C@@]2(CCN3CC2CC2)[C@H]1O5',
                         'CN(C)CCCN1c2ccccc2CCc2ccc(Cl)cc21',
                         'O=C(c1ccc(F)cc1)C1CCN(CCn2c(=O)[nH]c3ccccc3c2=O)CC1',
                         'CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1',
                          'O=C1O[C@H]([C@@H](O)CO)C(O)=C1O'])
    property_dict = get_properties(population, args)
    print(property_dict)


