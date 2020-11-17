import argparse
arg_lists = []
parser = argparse.ArgumentParser(description='GreetingsClass')

def add_argument_grp(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

####### List all your inputs ###########
data_args = add_argument_grp(name='rsics dataset')
data_args.add_argument("--csv", type=str,
                       default='rsics_dataset/tagged_selections_by_sentence.csv',
                       help='dataset file path')
data_args.add_argument('--vocalb_size', type=int, default=25000)
train_args = add_argument_grp(name='training_args')
train_args.add_argument("--epochs", type=int,
                       default=20,
                       help='set number of epochs to train the model as --epochs 20')

train_args.add_argument("--lr", type=int,
                       default=0.0001,
                       help='learning rate')

model_args = add_argument_grp(name='model_args')

model_args.add_argument("--embedding_dim", type=int,
                       default=200,
                       help='dim of embedding vector')
model_args.add_argument("--hidden_dim", type=int,
                       default=256,
                       help='number of LSTM units')
model_args.add_argument("--out_dim", type=int,
                       default=1,
                       help='based on number of classes set out_dim')
model_args.add_argument("--n_layers", type=int,
                       default=3,
                       help='set total layers')
model_args.add_argument("--dropout", type=int,
                       default=0.2)
model_args.add_argument("--batch_size", type=int,
                       default=128)

logs_args = add_argument_grp(name='logs_args')

logs_args.add_argument('--log_dir', type=str, default='logs')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed