from long_sequence_tools import *
from highlight_utils import *
from sts_tools import get_sbert_embeddings

def get_pio_abstracts(abstracts, model, tokenizer, model_sbert):
  with torch.no_grad():
    model.eval()
    model_sbert.eval()
    inputs = tokenizer(abstracts, return_tensors='np', max_length = 512, padding = 'max_length', return_offsets_mapping = True)

    input_ids = inputs['input_ids']
    offset_maps = inputs['offset_mapping']

    abstract_lengths = [np.max(np.nonzero(input_ids[i])) + 1 for i in range(len(abstracts))]
    inputs_split, original_sentence_mapping, all_paragraph_start_indices = split_long_sequences(input_ids)
    

    inputs_split['input_ids'] = inputs_split['input_ids'].to('cuda:0')
    inputs_split['token_type_ids'] = inputs_split['token_type_ids'].to('cuda:0')
    inputs_split['attention_mask'] = inputs_split['attention_mask'].to('cuda:0')

    outputs = model(**inputs_split)['logits']

    all_probs = torch.softmax(outputs, 2).detach().cpu().numpy()
    all_probs = merge_outputs(all_probs, original_sentence_mapping, all_paragraph_start_indices)
    all_highlight_indices = []

    pio_counts = []
    all_tokens = []
    attention_masks = []
    all_abstract_blocks = []

    for i in range(len(abstracts)):
      probs = all_probs[i][:abstract_lengths[i]]
      offset_map = offset_maps[i]
      sequence_tokens = input_ids[i][1:abstract_lengths[i]]
      tokens, attention_mask, pio_count, abstract_blocks = get_pio_block_tokens(probs, [[0,3], [1,4], [2,5], [], [], [], [6]], 6, sequence_tokens, model_sbert, offset_map = offset_map)
      all_tokens.append(tokens)
      attention_masks.append(attention_mask)
      pio_counts.append(pio_count)
      all_abstract_blocks.append(abstract_blocks)
        
    inputs = {
      'input_ids' : zero_cat(all_tokens),
      'attention_mask': zero_cat(attention_masks)
    }

    embeddings = get_sbert_embeddings(inputs, model, tokenizer, as_tokens = True)

    split_embeddings = resplit_embeddings(embeddings, pio_counts)


    for abstract_pio_embeddings, pio_count, abstract_blocks, offset_map in zip(split_embeddings, pio_counts, all_abstract_blocks, offset_maps):
      highlight_indices = get_deduplicated_blocks_from_embeddings(abstract_blocks, offset_map, abstract_pio_embeddings, pio_count)
      all_highlight_indices.append(highlight_indices)

    return all_highlight_indices

def resplit_embeddings(embeddings, pio_counts):
  split_embeddings = []
  offset = 0
  
  for pio_count in pio_counts:
    split_embeddings.append([])
    for attribute_count in pio_count:
      split_embeddings[-1].append(embeddings[offset:offset+attribute_count])
      offset += attribute_count

  return split_embeddings
  