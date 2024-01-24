from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS
import re
import copy
import numpy as np
import re
import torch

from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')

import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import torch.nn as nn

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

app = Flask(__name__)
CORS(app)

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '>',  '♥', '←', '§', '″', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '-', '●', 'â', '►', '-', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '<', '─',
 '▒', ':', '¼', '⊕', '▼', '▪', '†', '■', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', ',', '¾', 'Ã', '⋅', '∞',
 '∙', ')', '↓', '、', '│', '()', '»', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot",
                "'cause": "because", "could've": "could have",
                "couldn't": "could not", "didn't": "did not",
                "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did",
                "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have","I'm": "I am", "I've": "I have",
                "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                "i'll've": "i will have","i'm": "i am", "i've": "i have",
                "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                "it'll": "it will", "it'll've": "it will have","it's": "it is",
                "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                "might've": "might have","mightn't": "might not",
                "mightn't've": "might not have", "must've": "must have",
                "mustn't": "must not", "mustn't've": "must not have",
                "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not",
                "oughtn't've": "ought not have", "shan't": "shall not",
                "sha'n't": "shall not", "shan't've": "shall not have",
                "she'd": "she would", "she'd've": "she would have",
                "she'll": "she will", "she'll've": "she will have",
                "she's": "she is", "should've": "should have",
                "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have","so's": "so as", "this's": "this is",
                "that'd": "that would", "that'd've": "that would have",
                "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is",
                "here's": "here is","they'd": "they would",
                "they'd've": "they would have", "they'll": "they will",
                "they'll've": "they will have", "they're": "they are",
                "they've": "they have", "to've": "to have",
                "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are",
                "what's": "what is", "what've": "what have", "when's": "when is",
                "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will",
                "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have",
                "will've": "will have", "won't": "will not", "won't've": "will not have",
                "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all",
                "y'all'd": "you all would","y'all'd've": "you all would have",
                "y'all're": "you all are","y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have",
                "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color',
                'centre': 'center', 'favourite': 'favorite',
                'travelling': 'traveling', 'counselling': 'counseling',
                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist',
                'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does'}

embed_size = 200
max_features = 99001
maxlen = 70
batch_size = 512
n_epochs = 5
n_splits = 5
SEED = 10
debug = 0


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def load_tokenizer_config():
    with open('tokenizer_config.pkl', 'rb') as config_file:
        config = pickle.load(config_file)
    return config['word_index']

def preprocess_text(text, tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=maxlen)
    return padded_sequence

word_index = load_tokenizer_config()

tokenizer = Tokenizer(num_words=max_features)
tokenizer.word_index = word_index

glove_embeddings = np.load('embedding_matrix.npy')

class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
    def __getitem__(self,index):
        data,target = self.dataset[index]
        return data,target,index
    def __len__(self):
        return len(self.dataset)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class BiLSTM_single(nn.Module):

    def __init__(self):
        super(BiLSTM_single, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(glove_embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.unsqueeze(h_embedding, 0)
        h_embedding = torch.squeeze(h_embedding, 1)

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)

        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
    
def load_and_predict(x_test, model_obj):
    model = copy.deepcopy(model_obj)

    model.load_state_dict(torch.load('your_model_weights.pth', map_location=torch.device("cpu")))

    x_test_tensor = torch.tensor(x_test, dtype=torch.long).cpu()

    with torch.no_grad():
        y_pred = model(x_test_tensor).detach()

    predicted_prob = torch.sigmoid(y_pred).cpu().numpy()[:, 0]
    predicted_class = (predicted_prob > 0.5).astype(int)

    return predicted_prob, predicted_class

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data["text"]
    if text is not None:
        text = text.lower()
        text = clean_text(text)
        text = clean_numbers(text)
        text = replace_typical_misspell(text)
        text = preprocess_text(text, tokenizer, maxlen)

        predicted_prob, predicted_class = load_and_predict(text, BiLSTM_single())
        
        return jsonify({'prediction': int(predicted_class[0])})
    else:
        return jsonify({'error': 'Input text not provided.'})

if __name__ == '__main__':
    app.run(debug=True)
