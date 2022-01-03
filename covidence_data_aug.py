import csv
import json
import numpy as np
import random
import re
from abstract_prep import prepare_abstract

def generate_splits(seed):
		test_titles = {}

		test_csv = open('./csvs_for_training/abstracts_for_review.csv', encoding="utf8")
		field_names = ("Title","Authors","Abstract")
		reader = csv.DictReader(test_csv, field_names)

		for i, row in enumerate(reader):
			if not row['Title']:
				continue
			test_titles[row['Title']] = False

		np.random.seed(100)
		test_allocations = []
		for i in range(80000):
				allocation = np.random.choice([0, 1], p=[0.7, .3])    
				test_allocations.append(allocation)	

		np.random.seed(seed) 
		allocations = []
		for i in range(80000):
				allocation = np.random.choice([0, 1, 2], p=[0.4, .6, .0])    
				allocations.append(allocation)
		# print(allocations[0:100])

		train_file = open('./data/text_classification/covidence_ensemble{}/train.tsv'.format(seed), 'w', encoding="utf8")
		dev_file = open('./data/text_classification/covidence_ensemble{}/dev.tsv'.format(seed), 'w', encoding="utf8")
		test_file = open('./data/text_classification/covidence_ensemble{}/test.tsv'.format(seed), 'w', encoding="utf8")
		stage_file = open('./data/text_classification/covidence_ensemble{}/stage.tsv'.format(seed), 'w', encoding="utf8")
		combined_file = open('./data/text_classification/covidence_ensemble{}/combined.tsv'.format(seed), 'w', encoding="utf8")

		train_writer = csv.writer(train_file, delimiter='\t')
		dev_writer = csv.writer(dev_file, delimiter='\t')
		test_writer = csv.writer(test_file, delimiter='\t')
		stage_writer = csv.writer(stage_file, delimiter='\t')
		combined_writer = csv.writer(combined_file, delimiter='\t')

		train_writer.writerow(['guid', 'sentence', 'label'])
		dev_writer.writerow(['guid', 'sentence', 'label'])
		test_writer.writerow(['guid', 'sentence', 'label'])
		stage_writer.writerow(['guid', 'sentence', 'label'])
		combined_writer.writerow(['guid', 'sentence', 'label'])

		file_names = ['included', 'excluded', 'irrelevant']
		labels = ['included', 'false', 'false']
		field_names = ("Title","Authors","Abstract")
		current_index = 0
		dong_count = 0

		test_count = 0
		for file_name, label in zip(file_names, labels):
				csvfile = open('./csvs_for_training/' + file_name + '.csv', 'r', encoding = 'utf8')

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
							writers = [train_writer, dev_writer, test_writer, stage_writer]

							if(test_allocations[dong_count] == 1):
								writer_index = 2
							else:
								writer_index = allocations[dong_count]

							if(title in test_titles):
								writer_index = 3 #if it is in the special abstracts, put it in stage
								if (test_titles[title]):
									continue
								test_titles[title] = file_name
								test_count += 1

							writer = writers[writer_index]
							writer.writerow([0, text, label])
							if label == 'included' and writer_index == 0:
								for j in range(24):
									writer.writerow([0, text, label])

							if writer_index != 0:
								combined_writer.writerow([0, text, label])
						dong_count += 1
				csvfile.close()

		print(dong_count)

		train_file.close()
		dev_file.close()
		test_file.close()

if __name__ == '__main__':
		for i in range(5):
				generate_splits(i)