import csv
import json
import numpy as np

if __name__ == '__main__':
    np.random.seed(0) 
    train_file = open('./abstract_screen_demo/data/text_classification/covidence/train.tsv', 'w')
    dev_file = open('./abstract_screen_demo/data/text_classification/covidence/dev.tsv', 'w')
    test_file = open('./abstract_screen_demo/data/text_classification/covidence/test.tsv', 'w')

    train_writer = csv.writer(train_file, delimiter='\t')
    dev_writer = csv.writer(dev_file, delimiter='\t')
    test_writer = csv.writer(test_file, delimiter='\t')

    train_writer.writerow(['guid', 'sentence', 'label'])
    dev_writer.writerow(['guid', 'sentence', 'label'])
    test_writer.writerow(['guid', 'sentence', 'label'])

    file_names = ['included', 'excluded', 'irrelevant']
    labels = ['included', 'false', 'false']
    field_names = ("Title","Authors","Abstract")

    dong_count = 0
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
            
            writer = np.random.choice([train_writer, dev_writer, test_writer], p=[0.7, .1, .2])
            
            writer.writerow([0, text, label])
            dong_count += 1
        csvfile.close()

    print(dong_count)

    train_file.close()
    dev_file.close()
    test_file.close()