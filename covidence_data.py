import csv
import json
import numpy as np

if __name__ == '__main__':
    np.random.seed(0) 
    train_file = open('./abstract_screen_demo/data/text_classification/covidence/train.tsv', 'w')
    dev_file = open('./abstract_screen_demo/data/text_classification/covidence/dev.tsv', 'w')
    test_file = open('./abstract_screen_demo/data/text_classification/covidence/test.tsv', 'w')

    file_names = ['included', 'excluded', 'irrelevant']
    labels = ['included', 'excluded', 'excluded']
    field_names = ("Title","Authors","Abstract")

    for file_name, label in zip(file_names, labels):
        csvfile = open('./abstract_screen_demo/data/text_classification/covidence/' + file_name + '.csv', 'r')

        reader = csv.DictReader( csvfile, field_names)
        first_line = True
        for row in reader:
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

            new_dict = {'text': text, 'label' : label, 'metadata':''}
            
            output_file = np.random.choice([train_file, dev_file, test_file], p=[0.7, .1, .2])
            
            json.dump(new_dict, output_file)
            output_file.write('\n')
        
        csvfile.close()

    train_file.close()
    dev_file.close()
    test_file.close()