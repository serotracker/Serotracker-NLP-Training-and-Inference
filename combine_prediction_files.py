from abstract_prep import prepare_abstract
import csv
import hashlib
import argparse

#for combining the inclusion recommendations and the PIO predictions into a single file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    csvs = args.files
    print(csvs)

    prediction_filenames = ['./output/covidence_ensemble0/all_predictions.txt', './output/covidence_ensemble1/all_predictions.txt', './output/covidence_ensemble2/all_predictions.txt', './output/covidence_ensemble3/all_predictions.txt', './output/covidence_ensemble4/all_predictions.txt']
    prediction_lines = [open(prediction_filename, 'r', encoding="utf8").readlines() for prediction_filename in prediction_filenames]
    pio_file_lines = open('./output/pio_predictions.txt', 'r', encoding="utf8").read().splitlines()
    output_file = open('./output/all_predictions.txt', 'w', encoding="utf8")

    output_writer = csv.writer(output_file, delimiter='\t')

    output_writer.writerow(['hash', 'prediction', 'pio'])

    texts_with_abstract_count = 0

    for csv_filename in csvs:
      csvfile = open(csv_filename, 'r', encoding="utf8")


      field_names = ("Title","Authors","Abstract")
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
          text = prepare_abstract(title, '')

          
          if len(abstract) > 0:
            prediction = 0
            #average over the 5 predictions
            for e in range(5):
              prediction += float(prediction_lines[e][texts_with_abstract_count].split()[3])
            prediction = prediction/5
            pio_tags = pio_file_lines[texts_with_abstract_count]
            hashed_title = str(hashlib.sha256(text.encode('utf-8')).hexdigest())
            output_writer.writerow([hashed_title, prediction, pio_tags])
            texts_with_abstract_count += 1

      csvfile.close()

    output_file.close()