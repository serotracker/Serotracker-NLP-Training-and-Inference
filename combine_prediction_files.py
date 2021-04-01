from abstract_prep import prepare_abstract
import csv
import hashlib

if __name__ == '__main__':
    original_csv_filename = './abstract_screen_demo/data/text_classification/covidence/' + 'all' + '.csv'
    prediction_filenames = ['./output/covidence_aug_mi2/all_predictions.txt']
    prediction_lines = [open(prediction_filename, 'r').readlines() for prediction_filename in prediction_filenames]
    pio_file_lines = open('./pio_predictions.txt', 'r').read().splitlines()
    output_file = open('./all_predictions.txt', 'w')

    output_writer = csv.writer(output_file, delimiter='\t')

    output_writer.writerow(['hash', 'prediction', 'pio'])

    csvfile = open(original_csv_filename, 'r')


    field_names = ("Title","Authors","Abstract")
    reader = csv.DictReader( csvfile, field_names)
    first_line = True
    texts_with_abstract_count = 0
    for i, row in enumerate(reader):
        if first_line:
            first_line = False
            continue
        if not row['Title']:
            continue

        title = row['Title']
        abstract = row['Abstract']
        text = prepare_abstract(title, '')


        if len(abstract) > 0:
          prediction = prediction_lines[0][texts_with_abstract_count].split()[3]
          pio_tags = pio_file_lines[texts_with_abstract_count]
          hashed_title = str(hashlib.sha256(text.encode('utf-8')).hexdigest())
          output_writer.writerow([hashed_title, prediction, pio_tags])
          texts_with_abstract_count += 1

    csvfile.close()

    output_file.close()