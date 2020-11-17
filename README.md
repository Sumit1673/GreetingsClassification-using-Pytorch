# GreetingsClassification-using-Pytorch
# Info about the dataset
Greetings classification is based on **Relational Stratigies in Customer Service ([RSiCS](https://nextit-public.s3-us-west-2.amazonaws.com/rsics.html)) Dataset**. This dataset is created by allowing humans to interact with **Intelligent Virtual Agents** in the buisness
of travel. Common resource are Airline Forums such as TripAdvisor.

Source of this dataset:
1. TripAdvisor.com airline forum
2. Train travel IVA
3. Airline travel IVA
4. Telecommunications support IVA
Basically the dataset used for this project captures the emotions in terms of greetings, ranting and others and the goal of this project focuses on 
predicting the greetings in the message. In the dataset description it its given to use the field 'Selected' for any type of classification.

To use this code one can download the dataset from the above URL and use '[tagged_selections_by_sentence.csv](https://nextit-public.s3-us-west-2.amazonaws.com/rsics.html#tagged95selections95by95sentencecsv)'. This is only the file which contains Greetings and many more emotions.

# Pre-requisite: Python=3.7
1. PyTorch=1.6
2. TorchText=0.6
3. spacy (for tokenizing)
  - To install spacey create a venv 
  - do: pip install -U spacy
  - python -m spacy download en
  
4. GloVE is used for vocalbulary this is will be installed automatically from the code.
5. Streamlit for basic visualization of the data.

# Folder Structure
├── config.py

├── dataloader.py

├── dataset.py 

├── engine.py 

├── __init__.py

├── main.py 

├── predict.py 

├── rsics_dataset 

├── runs 

├── trainer.py 



# How to run
Before running the main.py file go through the config.py file and check if any parameters you would like to adjust.

do: python main.py
