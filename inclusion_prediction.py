import torch
import numpy as np

def get_inclusion_likelihoods(abstract_dict, model, tokenizer):
    predictions_dict = {}
    merged_abstracts = []
    title_list = []
    for title, abstract in abstract_dict.items():
        # abstract_modified = abstract.replace('sars-cov-2', 'coronavirus')
        title_list.append(title)
        if title[-1] != '.':
            title = title + '.'
        text = title + ' ' + abstract
        
        merged_abstracts.append(text)

    inputs = tokenizer(merged_abstracts, max_length = 512, return_tensors = 'pt', padding=True, truncation = True)

    inputs['input_ids'] = inputs['input_ids'].to('cuda:0')
    inputs['token_type_ids'] = inputs['token_type_ids'].to('cuda:0')
    inputs['attention_mask'] = inputs['attention_mask'].to('cuda:0')

    with torch.no_grad():
        model.eval()
        logits = model(**inputs)['logits']
        predictions = torch.softmax(logits, 1).detach().cpu().numpy()

    for i, title in enumerate(title_list):
        predictions_dict[title] = predictions[i]

    return predictions_dict