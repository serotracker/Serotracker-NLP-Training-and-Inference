import csv

if __name__ == '__main__':
	train_file = open('./abstract_screen_demo/data/text_classification/covidence2/train.tsv', 'w')

	train_file.write('guid\tsentence\tlabel\n')
	base_train_file = open('./abstract_screen_demo/data/text_classification/covidence/dev.tsv', 'r')
	preds_files = open('./output/covidence/dev_predictions.txt', 'r')
	lines = base_train_file.readlines()
	pred_lines = preds_files.readlines()
    

	for i, (line, pred_line) in enumerate(zip(lines[1:], pred_lines)):
		train_file.write(line)
		things = pred_line.split('\t')
		# print(things)
		if things[0] != things[1]:
			
			if things[0] == 'included':
				for i in range(9):
					train_file.write(line)
			else:
				for i in range(9):
					train_file.write(line)

    
	train_file.close()