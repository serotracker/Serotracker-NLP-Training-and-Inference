from itertools import groupby
import numpy as np
import torch

PERIOD_TOKEN = 18
COLON_TOKEN = 30
START_TOKEN = 2
END_TOKEN = 3


def split_abstract(sequence_tokens):
  paragraph_splits = []
  paragraph_start_indices = [0]
  period_indices = []
  sentence_splits = []
  current_sentence = []
  for i, token in enumerate(sequence_tokens[1:]):
    current_sentence.append(token)
    if(token == PERIOD_TOKEN or token == COLON_TOKEN):
      sentence_splits.append(current_sentence)
      current_sentence = []
      period_indices.append(i + 1)
  period_indices = np.roll(period_indices, 1)
  current_length = 1
  current_sequence = [START_TOKEN]
  for sentence, period_index in zip(sentence_splits, period_indices):
    if current_length + len(sentence) + 1 > 512:
      new_sequence = current_sequence + [END_TOKEN]
      new_sequence = new_sequence + [0 for i in range(512 - len(new_sequence))]
      paragraph_splits.append(new_sequence)
      current_length = 1
      current_sequence = [START_TOKEN]
      paragraph_start_indices.append(period_index + 1)
    
    current_length += len(sentence)
    current_sequence.extend(sentence)

  new_sequence = current_sequence + [END_TOKEN]
  new_sequence = new_sequence + [0 for i in range(512 - len(new_sequence))]
  paragraph_splits.append(new_sequence)
  return paragraph_splits, paragraph_start_indices

def split_long_sequences(token_ids_long):
  original_sentence_mapping = []
  all_paragraph_start_indices = []
  all_token_splits = []
  for i, sequence_tokens in enumerate(token_ids_long):
    abstract_token_splits, paragraph_start_indices = split_abstract(sequence_tokens)
    original_sentence_mapping.extend([i for j in range(len(abstract_token_splits))])
    all_token_splits.extend(abstract_token_splits)
    all_paragraph_start_indices.extend(paragraph_start_indices)
  input_ids = np.array(all_token_splits)
  attention_mask = np.array(input_ids != 0) * 1
  token_type_ids = np.zeros_like(input_ids)

  # print(input_ids)      
  # for l in input_ids:
  #   print(len(l))

  inputs = {
      'input_ids': torch.from_numpy(input_ids).long(),
      'attention_mask': torch.from_numpy(attention_mask).long(),
      'token_type_ids': torch.from_numpy(token_type_ids).long()
  }
  return inputs, original_sentence_mapping, all_paragraph_start_indices

def merge_outputs(output, original_sentence_mapping, all_paragraph_start_indices):
  outputs_to_append = []
  all_outputs = []
  for i in range(len(original_sentence_mapping)):
    if i == 0:
      outputs_to_append.append(output[i])
    else:
      if original_sentence_mapping[i] != original_sentence_mapping[i - 1]:
        all_outputs.append(np.concatenate(outputs_to_append))
        outputs_to_append = [output[i]]
      else:
        #add current output to existing one
        #remove trailing shit 
        outputs_to_append[-1] = outputs_to_append[-1][0:all_paragraph_start_indices[i] - all_paragraph_start_indices[i-1]]
        outputs_to_append.append(output[i][1:])
  
  all_outputs.append(np.concatenate(outputs_to_append))

  return all_outputs
