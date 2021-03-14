import csv
import json
import numpy as np
import random
import re
from textaugment import Wordnet, Word2vec, Translate
import gensim

if __name__ == '__main__':
    np.random.seed(0) 
    allocations = []
    for i in range(40000):
        allocation = np.random.choice([0, 1, 2], p=[0.4, .4, .2])    
        allocations.append(allocation)
    # for i in range(40000):
    #   if allocations[i] != 2:
    #     # allocations[i] = 1- allocations[i]
    #     if allocations[i] == 0:
    #         c = np.random.choice([0, 1], p = [0.5, .5])
    #         allocations[i] = c
    print(allocations[0:100])
    random.seed(0)
    # t = Wordnet()
    t = Wordnet(v=True ,n=True, p=0.5)
    # model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True, limit=50000)
    # t = Word2vec(model=model)
    # t = Translate(src = 'en', to = 'fr')
    train_file = open('./abstract_screen_demo/data/text_classification/covidence_aug/train.tsv', 'w')
    dev_file = open('./abstract_screen_demo/data/text_classification/covidence_aug/dev.tsv', 'w')
    test_file = open('./abstract_screen_demo/data/text_classification/covidence_aug/test.tsv', 'w')

    train_writer = csv.writer(train_file, delimiter='\t')
    dev_writer = csv.writer(dev_file, delimiter='\t')
    test_writer = csv.writer(test_file, delimiter='\t')

    train_writer.writerow(['guid', 'sentence', 'label'])
    dev_writer.writerow(['guid', 'sentence', 'label'])
    test_writer.writerow(['guid', 'sentence', 'label'])

    file_names = ['included', 'excluded', 'irrelevant']
    labels = ['included', 'false', 'false']
    field_names = ("Title","Authors","Abstract")
    current_index = 0
    dong_count = 0
    for file_name, label in zip(file_names, labels):
        csvfile = open('./abstract_screen_demo/data/text_classification/covidence/' + file_name + '.csv', 'r')

        reader = csv.DictReader( csvfile, field_names)
        first_line = True
        for i, row in enumerate(reader):
            if first_line:
                first_line = False
                continue
            if not row['Title']:
                continue

            title = row['Title']
            abstract = row['Abstract']
            if title[-1] != '.':
                title = title + '.'
            text = title + ' ' + abstract
            text = text.replace('<h4>', ' ')
            text = text.replace('</h4>', ' ')
            
            if len(abstract) > 0:
              writers = [train_writer, dev_writer, test_writer]
              writer_index = allocations[dong_count]
              writer = writers[writer_index]
              writer.writerow([0, text, label])
              # if label == 'included' and writer_index == 0:
              #   for i in range(30):
              #     writer.writerow([0, text, label])
              # if writer_index == 0:
              #   print(i)
              #   for i in range(3):
              #     subtext = t.augment(text)
              #     # delimiter =  re.compile("(?<=[a-z])\.")
              #     # sentences = [x + '.' for x in delimiter.split(text)]
              #     # if len(sentences) > 4:
              #     #   j = random.randint(0, len(sentences) - 4)
              #     #   subtext = ' '.join(sentences[j:j+4])
              #     writer.writerow([0, subtext, label])
            dong_count += 1
        csvfile.close()

    print(dong_count)

    train_file.close()
    dev_file.close()
    test_file.close()