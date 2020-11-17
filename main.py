import dataset
import trainer
import streamlit as st
import numpy
import torch
from config import get_config
from dataloader import BuildDataset
import warnings
from engine import LSTMNet
warnings.filterwarnings("ignore")

config, _ = get_config()

st.title(' Greetings? Yes | No')


def main():

   menu = ["Visualize Data", "Test Model"]
   choice = st.sidebar.selectbox("Menu", menu)
   if choice == 'Visualize Data':
       st.success('Visualizing Data')
       visualize_data()
   else:
       test_data()

def visualize_data():
   data_dict = dataset.get_data()
   build_data = BuildDataset()
   train_ds, test_ds = build_data.get_dataset(data_dict['train_data'], data_dict['test_data'])

   train_iter, test_iter = build_data.create_vocalb(train_ds,
                                                    test_ds)

   st.subheader('How the data looks after creating vocalb from the dataset . . ')
   st.text(vars(test_ds[20]))
   print(vars(test_ds[20]))


   # create the model
   model = LSTMNet(config.vocalb_size, config.embedding_dim, input_dim=len(build_data.TEXT.vocab),
                   hidden_dim=config.hidden_dim, output_dim=config.out_dim,
                   n_layers=config.n_layers, dropout=config.dropout,
                   pad=build_data.TEXT.vocab.stoi[build_data.TEXT.pad_token])

   # Load pre-trained embedding weights
   pretrained_embeddings = build_data.TEXT.vocab.vectors

   print(pretrained_embeddings.shape)

   # model.embedding_layer.weight.data.copy_(pretrained_embeddings)

   #initialize to zeros
   model.embedding_layer.weight.data[model.pad_idx] = torch.zeros(config.embedding_dim)

   model_trained = trainer.train_net(model, train_iter, test_iter, epochs=1)
   # data = next(test_iter)
   # trainer.evaluate(model_trained, data)
   # import torchviz
   # torchviz.make_dot(out)

def test_data():
    st.subheader('Check Model Performance')
    inp_txt = st.text_input('Enter text')


if __name__ == '__main__':
    main()

