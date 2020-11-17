import torch
from torchtext import data
from config import get_config
import torch.nn as nn

config, _ = get_config()

SEED = 42
# for reproducibility
torch.manual_seed(SEED)
"""
torchtext data module is used to create FIELD Objects. These field objects will contain information
for converting the texts to Tensors. We will set two parameters:
    tokenize=spacy and
    include_arguments=True Which implies that SpaCy will be used to tokenize the texts and
        that the field objects should include length of the texts - which will be needed to pad the texts.
We will later use methods of these objects to create a vocabulary, which will help us create a
numerical representation for every token. The LabelField is a shallow wrappper around field, useful for
data labelling
[SOURCE]: https://gist.github.com/lextoumbourou/8f90313cbc3598ffbabeeaa1741a11c8

For Vocalb i used Glove: GloVe is an unsupervised learning algorithm for obtaining vector representations for words
https://nlp.stanford.edu/projects/glove/
"""

class RSICSDataset(data.Dataset):
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.Greeting if not is_test else None
            text = row.Selected
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

class BuildDataset:
    def __init__(self):
        self.TEXT = data.Field(tokenize='spacy', include_lengths=True)
        self.LABEL = data.LabelField(unk_token='UNK', dtype=torch.float, is_target=True)
        # self.LABEL = data.Field(unk_token=None, dtype=torch.int, is_target=True)

    def get_dataset(self, train_df, test_df):

        fields = [('text', self.TEXT), ('labels', self.LABEL)]

        self.train_ds, self.test_ds = RSICSDataset.splits(fields, train_df=train_df, test_df=test_df)
        return self.train_ds, self.test_ds

    def create_vocalb(self, train_ds, test_ds):
        if isinstance(train_ds, RSICSDataset):
            self.TEXT.build_vocab(train_ds,
                             max_size=config.vocalb_size,
                             vectors='glove.6B.200d',
                             unk_init=torch.Tensor.zero_)

            self.LABEL.build_vocab(train_ds)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            train_iterator, test_iterator = data.BucketIterator.splits(
                (train_ds, test_ds),
                batch_size=config.batch_size,
                sort_within_batch=True,
                device=device)
            return train_iterator, test_iterator

        else:
            print('Unknown dataset format. !!')


if __name__ == '__main__':
    build = BuildDataset()
    build.get_dataset()
