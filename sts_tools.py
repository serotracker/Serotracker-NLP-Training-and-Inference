import torch
import transformers
import collections
from abstract_screen_demo.utils.processors.utils_blue import blue_convert_examples_to_features
from abstract_screen_demo.utils.sbert_wrapper import get_sbert_embeddings
from sentence_transformers import util

def sentences_to_features(sentences1, sentences2, tokenizer):
  examples = []
  label_list = []
  Example = collections.namedtuple('Item', ('guid', 'label', 'text_a', 'text_b'))
  for s1, s2 in zip(sentences1, sentences2):
    examples.append(Example('0', 0, s1, s2))
    label_list.append(0)
  
  features = blue_convert_examples_to_features(examples,tokenizer,output_mode='regression', label_list = label_list, max_length=128)

  return features

def get_similarities(sentences1, sentences2, model, tokenizer):
  model.eval()
  features = sentences_to_features(sentences1, sentences2, tokenizer)

  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

  # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  model_input = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels, "token_type_ids": all_token_type_ids}
  print(model(**model_input)['logits'])


def get_similarities_sbert(sentences, model, tokenizer):
  model.eval()
  embeddings = get_sbert_embeddings(sentences, model, tokenizer)
  sims = util.pytorch_cos_sim(embeddings, embeddings)
  return sims