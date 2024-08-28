import argparse

def parse_arguments():
    all_columns = 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed,fub_spec,fub_spec_transformed,log_clearance,lipophilicity_az,clearance_az'
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--input_file_csv', type=str, default='this breaks',  help='location of input data csv file')
    parser.add_argument('--data_column', type=str, default='SMILES', help='data column in csv file')
    parser.add_argument('--score_columns', type=str, default=all_columns, help='comma separated score columns in csv file')
    parser.add_argument('--output_directory', type=str, default='./results/', help='output directory for model, predictions, and stats')
    parser.add_argument('--run_id', type=str, default='_run', help='run id for model')
    parser.add_argument('--tdc', action='store_true', default=False, help='whether this is a tdc run' )

    # data options
    parser.add_argument('--train_fraction', type=float, default=0.8, help='fraction of samples for training')
    parser.add_argument('--max_scale', type=float, default=-1.0, help='maximum used to divide scores')
    parser.add_argument('--number_randomized_smiles', type=int, default=0, help='option to add randomized smiles to training')
    parser.add_argument('--normalize', action='store_true', default=False, help='option to normalize data to mean 0 and standard deviation 1')
    parser.add_argument('--log_transform', action='store_true', default=False, help='option to apply log to training values')
    parser.add_argument('--split_column', type=str, default='split', help='what column in the csv to pull the split from')

    # pre-trained model information
    parser.add_argument('--tokenizer_name', type=str, default='/lustre/or-scratch/cades-ccsd/z6f/lombardo-ft/tokenizers/bert_tok', help='tokenizer from hugging face')
    parser.add_argument('--tokenizer_type', type=str, default='bert', help='type for pretokenizer')
    parser.add_argument('--model_name', type=str, default='/lustre/or-scratch/cades-ccsd/z6f/lombardo-ft/models/', help='model from hugging face')
    parser.add_argument('--use_random_weights', action='store_true', default=False, help='option to use random weights for pre-trained model')

    # training options
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--dropout_fraction', type=float, default=0.0, help='dropout fraction on fully-connected layer')
    parser.add_argument('--loss_type', type=str, default='MSE', choices=['MSE', 'BCEWithLogitsLoss'], help='type of loss function for regression')
    parser.add_argument('--validation_type', type=str, default='MSE', choices=['MSE', 'R2', 'Loss'], help='Metric to select best validation model')
    parser.add_argument('--device', type=str, default=None, help='device for training')
    parser.add_argument('--max_len', type=float, default=512, help='maximum sequence length')
    args = parser.parse_args()
    args.score_columns = args.score_columns.split(',')
    args.ntasks = len(args.score_columns)
    return args
