import torch
from torch.nn.functional import relu

from transformers import AutoModel
from transformers import AutoConfig

class Net(torch.nn.Module):
    def __init__(self, args, output_dim):
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
            return self.fc1(x)
        else:
            return self.fc1(x).squeeze(-1)

