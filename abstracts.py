from long_sequence_tools import *
from highlight_utils import *

def get_pio_abstracts(abstracts, model, tokenizer, model_sbert):
  with torch.no_grad():
    inputs = tokenizer(abstracts, return_tensors='np', max_length = 512, padding = 'max_length', return_offsets_mapping = True)

    input_ids = inputs['input_ids']
    offset_maps = inputs['offset_mapping']

    abstract_lengths = [np.max(np.nonzero(input_ids[i])) + 1 for i in range(len(abstracts))]
    inputs_split, original_sentence_mapping, all_paragraph_start_indices = split_long_sequences(input_ids)
    

    inputs_split['input_ids'] = inputs_split['input_ids'].to('cuda:0')
    inputs_split['token_type_ids'] = inputs_split['token_type_ids'].to('cuda:0')
    inputs_split['attention_mask'] = inputs_split['attention_mask'].to('cuda:0')
    # print("SDLFKJSDF")

    outputs = model(**inputs_split)['logits']

    all_probs = torch.softmax(outputs, 2).detach().cpu().numpy()
    all_probs = merge_outputs(all_probs, original_sentence_mapping, all_paragraph_start_indices)
    all_highlight_indices = []

    for i in range(len(abstracts)):
      probs = all_probs[i][:abstract_lengths[i]]
      offset_map = offset_maps[i]
      sequence_tokens = input_ids[i][1:abstract_lengths[i]]
      highlight_indices = blockify_probs_and_remove_duplicates(probs, [[0,3], [1,4], [2,5], [], [], [], [6]], 3, sequence_tokens, model_sbert, offset_map = offset_map)
      all_highlight_indices.append(highlight_indices)

    return all_highlight_indices