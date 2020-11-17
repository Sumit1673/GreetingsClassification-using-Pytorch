import csv
import numpy as np
import pandas as pd
from config import get_config
import streamlit as st

fields = ['Dataset','Partition', 'SentenceID', 'Threshold', 'MergedSelections', 'Unselected', 'Selected',
          'Greeting', 'Backstory', 'Justification',	'Rant',	'Gratitude', 'Other', 'Express Emotion']
# load data
def load_data(file_path, num_row=None):
    csv_df = pd.read_csv(file_path, delimiter=',', header=None, skipinitialspace=True,
                         skiprows=1, names=fields, usecols=fields)

    return csv_df

def get_data():
    config, unparsed = get_config()
    data_df = load_data(config.csv)

    clean_data = post_data(data_df)

    return split_data(clean_data)


def post_data(data_df):
    st.subheader('Raw Data: RSICS Dataset. Training Data count: ' + str(len(data_df)))
    st.text('Expand to view ')
    st.write(data_df)

    st.subheader('Data needed for our application')
    df_greetings = data_df[['Selected', "Greeting"]]
    for i in range(len(df_greetings['Greeting'])):
        if df_greetings.loc[i, ['Greeting']][0] != '0' and df_greetings.loc[i, ['Greeting']][0] != '1':
            df_greetings.at[i, 'Greeting'] = 10

    # drop all the nan and empty values from the column
    df_greetings = df_greetings[df_greetings['Selected'].notna()]
    df_greetings = df_greetings.astype({'Greeting': int})
    df_greetings = df_greetings.loc[df_greetings['Greeting'].isin([0, 1])]

    df_greetings['Selected'] = normalise_text(df_greetings["Selected"])
    st.subheader('Total Dataset after removing unnecessary characters: {}'.format(len(df_greetings['Selected'])))
    st.write(df_greetings)
    st.subheader('Lets find out how many sentences has a greeting and how many do not!')

    with_greet_hist = np.histogram(df_greetings['Greeting'], bins=[0, 1, 2])
    st.subheader('Dataset with Greetings (1): {} and without(0): {}'.format(with_greet_hist[0][0],
                                                                            with_greet_hist[0][1]))

    st.bar_chart(pd.DataFrame({'0': [with_greet_hist[0][0]], '1': [with_greet_hist[0][1]]}))

    return df_greetings
    # df_greetings['reviews_length']=df_greetings['Selected'].apply(len)
    # hist_review_len = np.histogram(df_greetings['reviews_length'])
    # st.bar_chart(hist_review_len)

# to clean data
def normalise_text (data_df):
    data_df = data_df.str.lower() # lowercase
    # data_df = data_df.str.replace(r"\#", "") # replaces hashtags
    # data_df = data_df.str.replace(r"http\S+", "URL")  # remove URL addresses
    # data_df = data_df.str.replace(r"@", "")
    # data_df = data_df.str.replace(r"-----", "")
    # data_df = data_df.str.replace(r"....", " ")
    # data_df = data_df.str.replace(r":\)", " ")
    data_df = data_df.str.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    # data_df = data_df.str.replace("\s{2,}", " ")
    return data_df

def split_data(data_df, test_per=0.2):
    total_len = len(data_df)
    test_data = data_df.loc[0:total_len*test_per, :].sample(frac=1)
    train_data = data_df.loc[total_len*test_per:, :].sample(frac=1)

    st.subheader('Training Data')
    with_greet_hist = np.histogram(train_data['Greeting'], bins=[0, 1, 2])
    st.subheader('Dataset with Greetings (1): {} and without(0): {}'.format(with_greet_hist[0][0],
                                                                            with_greet_hist[0][1]))
    st.bar_chart(pd.DataFrame({'0': [with_greet_hist[0][0]], '1': [with_greet_hist[0][1]]}))

    st.subheader('Test Data')
    with_greet_hist = np.histogram(test_data['Greeting'], bins=[0, 1, 2])
    st.subheader('Dataset with Greetings (1): {} and without(0): {}'.format(with_greet_hist[0][0],
                                                                            with_greet_hist[0][1]))
    st.bar_chart(pd.DataFrame({'0': [with_greet_hist[0][0]], '1': [with_greet_hist[0][1]]}))

    return {'train_data': train_data,
        'test_data': test_data
    }

if __name__ =='__main__':
    config, unparsed = get_config()
    get_data()
