import torch
from sentence_transformers import util

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_sbert_embeddings(sentences, model, tokenizer, as_tokens = False):
  #Tokenize sentences
  if not as_tokens:
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(model.device)
  else:
    encoded_input = sentences
  #Compute token embeddings
  model_output = model(**encoded_input)

  #Perform pooling. In this case, mean pooling
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  return sentence_embeddings

def get_train_loss(model, tokenizer, batch, scale = 4):
  joined_sentences = batch['s1'] + batch['s2']
  embeddings = get_sbert_embeddings(joined_sentences, model, tokenizer)
  embeddings1 = embeddings[:len(batch['s1'])]
  embeddings2 = embeddings[len(batch['s1']):]
  output = torch.diagonal((model.scale * util.pytorch_cos_sim(embeddings1, embeddings2) + model.shift))
  loss = torch.mean((output - batch['label'])**2)
  return loss, output, [embeddings1, embeddings2]