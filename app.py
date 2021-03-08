from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from pathlib import Path
from highlight_utils import *
from sample_abstracts import * 
from sts_tools import get_similarities_sbert, get_similarities
from abstracts import get_pio_abstracts
import time
import pprint
from numpyencoder import NumpyEncoder
from inclusion_prediction import get_inclusion_likelihoods

app = Flask(__name__)
cors = CORS(app)
app.json_encoder = NumpyEncoder

app.config['CORS_HEADERS'] = 'Content-Type'

tokenizer_pio = AutoTokenizer.from_pretrained("output/BC5CDR")
model_pio = AutoModelForTokenClassification.from_pretrained("output/BC5CDR")

model_pio.to('cuda:0')

tokenizer_sbert = AutoTokenizer.from_pretrained("output/medsts")
model_sbert = AutoModel.from_pretrained("output/medsts")

tokenizer_inclusion = AutoTokenizer.from_pretrained("output/covidence")
model_inclusion = AutoModelForSequenceClassification.from_pretrained("output/covidence")

model_inclusion.to('cuda:0')

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/abstract_screen', methods = ['POST'])
def abstract_screen():
    # print(request.json['abstracts'])
    titles = []
    abstracts = []
    empty_titles = []
    abstract_dict = request.json['abstracts']
    for key, value in abstract_dict.items():
        if(len(value) > 0):
            titles.append(key)
            abstracts.append(value)
        else:
            empty_titles.append(key)

    highlight_indices = get_pio_abstracts(abstracts, model_pio, tokenizer_pio, model_sbert.to('cuda:0'))
    
    output_dict = {}
    for i in range(len(titles)):
        output_dict[titles[i]] = highlight_indices[i]
    for i in range(len(empty_titles)):
        output_dict[empty_titles[i]] = [[],[],[]]
    # pprint.pprint(output_dict)

    predictions_dict = get_inclusion_likelihoods(abstract_dict, model_inclusion, tokenizer_inclusion)

    return {'highlights' : output_dict, 'predictions' : predictions_dict}