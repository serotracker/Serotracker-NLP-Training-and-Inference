from abstract_prep import prepare_abstract
import csv

if __name__ == '__main__':
    all_file = open('./abstract_screen_demo/data/text_classification/covidence_aug/all.tsv', 'w')

    all_writer = csv.writer(all_file, delimiter='\t')

    all_writer.writerow(['guid', 'sentence', 'label'])

    csvs = ['./excluded.csv',
    './included.csv',
    './irrelevant.csv',
    './screen.csv',
    './select.csv'
    ]

    for csv_filename in csvs:
      csvfile = open(csv_filename, 'r')
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
          text = prepare_abstract(title, abstract)


          if len(abstract) > 0:
            all_writer.writerow([0, text, "included"])

      csvfile.close()

    all_file.close()