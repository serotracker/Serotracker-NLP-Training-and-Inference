from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModel
from pathlib import Path
from highlight_utils import *
from sts_tools import get_similarities_sbert, get_similarities
from abstracts import get_pio_abstracts
import time
import pprint
from inclusion_prediction import get_inclusion_likelihoods
import csv
from abstract_prep import prepare_abstract
from tqdm import tqdm
import argparse

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    csvs = args.files

    abstracts = []
    titles = []

    for csv_filename in csvs:
      csvfile = open(csv_filename, encoding="utf8")
      field_names = ("Title","Authors","Abstract")
      reader = csv.DictReader( csvfile, field_names)
      first_line = True
      
      for i, row in enumerate(reader):
          #make sure it's note the first line and not an empty line
          if first_line:
              first_line = False
              continue
          if not row['Title']:
              continue
        
            #check for title, then add it to list after preprocessing
          abstract = row['Abstract']
          title = row['Title']
          if len(abstract) > 0:
            abstract = prepare_abstract('', abstract)
            abstracts.append(abstract)
            titles.append(prepare_abstract(title, ''))

      csvfile.close()


    #load pretrained tokenizer and model train on PIO task - don't ask why it is BC5CDR
    tokenizer_pio = AutoTokenizer.from_pretrained("output/BC5CDR")
    tokenizer_pio.add_tokens(["&middot;"])
    model_pio = AutoModelForTokenClassification.from_pretrained("output/BC5CDR")
    model_pio.resize_token_embeddings(len(tokenizer_pio))

    model_pio.to('cuda:0')

    #load pretrained model for semantic similarity
    tokenizer_sbert = AutoTokenizer.from_pretrained("output/medsts")
    tokenizer_sbert.add_tokens(["&middot;"])
    model_sbert = AutoModel.from_pretrained("output/medsts")
    model_sbert.resize_token_embeddings(len(tokenizer_sbert))

    output_file = open('./output/pio_predictions.txt', 'w', encoding="utf8")
    i = 0

    for abstract_batch, title_batch in tqdm(zip(batch(abstracts, 16), batch(titles, 16))):
        #batch the abstracts into groups of 16 and get the predictions for each
        highlight_indices = get_pio_abstracts(abstract_batch, model_pio, tokenizer_pio, model_sbert.to('cuda:0'))
        for highlights in highlight_indices:
            output_file.write(str(highlights) + '\n')

    output_file.close()
        