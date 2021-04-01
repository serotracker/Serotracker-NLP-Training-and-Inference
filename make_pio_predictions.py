from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from pathlib import Path
from highlight_utils import *
from sample_abstracts import * 
from sts_tools import get_similarities_sbert, get_similarities
from abstracts import get_pio_abstracts
import time
import pprint
from inclusion_prediction import get_inclusion_likelihoods
import csv
from abstract_prep import prepare_abstract

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == '__main__':
    csvfile = open('./abstract_screen_demo/data/text_classification/covidence/' + 'all' + '.csv', 'r')
    field_names = ("Title","Authors","Abstract")
    reader = csv.DictReader( csvfile, field_names)
    first_line = True
    abstracts = []
    titles = []
    for i, row in enumerate(reader):
        if first_line:
            first_line = False
            continue
        if not row['Title']:
            continue

        abstract = row['Abstract']
        title = row['Title']
        if len(abstract) > 0:
          abstract = prepare_abstract('', abstract)
          abstracts.append(abstract)
          titles.append(prepare_abstract(title, ''))

    csvfile.close()

    tokenizer_pio = AutoTokenizer.from_pretrained("output/BC5CDR")
    model_pio = AutoModelForTokenClassification.from_pretrained("output/BC5CDR")

    model_pio.to('cuda:0')

    tokenizer_sbert = AutoTokenizer.from_pretrained("output/medsts")
    model_sbert = AutoModel.from_pretrained("output/medsts")

    output_file = open('./pio_predictions.txt', 'w')

    for abstract_batch, title_batch in zip(batch(abstracts, 16), batch(titles, 16)):
        highlight_indices = get_pio_abstracts(abstract_batch, model_pio, tokenizer_pio, model_sbert.to('cuda:0'))
        for highlights in highlight_indices:
            output_file.write(str(highlights) + '\n')

    output_file.close()
        