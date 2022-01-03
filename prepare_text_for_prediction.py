from abstract_prep import prepare_abstract
import csv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument(
        "--output_dir",
        default='covidence_aug',
        type=str,
        help="sldkfjslkdf",
    )
    args = parser.parse_args()

    all_file = open('./data/text_classification/{}/all.tsv'.format(args.output_dir), 'w', encoding="utf8")

    all_writer = csv.writer(all_file, delimiter='\t')

    all_writer.writerow(['guid', 'sentence', 'label'])

    csvs = args.files
    print(csvs)

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
          text = prepare_abstract(title, abstract)


          if len(abstract) > 0:
            all_writer.writerow([0, text, "included"])

      csvfile.close()

    all_file.close()