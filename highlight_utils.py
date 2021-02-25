import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import convolve1d
import numpy as np
import torch
from abstract_screen_demo.sts_tools import get_similarities_sbert

def modify_text(text):
  if text == '[CLS]':
    return ''
  if text.startswith('##'):
    return text[2:]
  if text == '[SEP]':
    return ''
  if text in ['.', ',',')', ';', ':']:
    return text
  return ' ' + text

def prob_to_color(prob):
  #I, O, P
  # return '#%02x%02x%02x' % (int((prob[0] + prob [3]) * 255), int((prob[1] + prob [4]) * 255), int((prob[2] + prob [5]) * 255))
  return '#%02x%02x%02x' % (int((prob[1] + prob [4]) * 255), int((prob[2] + prob [5]) * 255), int((prob[0] + prob [3]) * 255))

def highlight_text(tokens, predictions, max_percents, label_list, prediction_list, color_list, class_names, probs = None, threshold = None):
  figure = plt.figure(dpi = 100, figsize = [15, 0.6])
  x_position = 0
  y_position = 0
  plt.axis('off')
  
  for i, (token, prediction, conf) in enumerate(zip(tokens, predictions, max_percents)):
    prediction_name = label_list[prediction]
    colored = False
    text = modify_text(token)
    if probs is None:
      for i, (prediction_type, color) in enumerate(zip(prediction_list, color_list)):
        if prediction_type in prediction_name and (threshold is None or conf > threshold):
          alpha = conf * 0.5
          if 'B-' in prediction_type:
            alpha = conf * 1.0
          text = plt.text(x_position, y_position, text, bbox={'facecolor': color, 'alpha': alpha, 'pad': 0, 'edgecolor':'none'}, fontsize = 14)
          colored = True

      if not colored:
        text = plt.text(x_position, y_position, text, fontsize = 14)
    else:
      text = plt.text(x_position, y_position, text, bbox={'facecolor': prob_to_color(probs[i]), 'alpha': 0.5, 'pad': 0, 'edgecolor':'none'}, fontsize = 14)
    # transf = plt.transData.inverted()
    transf = plt.gca().transData.inverted()
    bb = text.get_window_extent(renderer = figure.canvas.get_renderer())
    bb_datacoords = bb.transformed(transf)
    x_position = bb_datacoords.x1 + 0.000
    if x_position > 0.8:
      y_position -= (bb_datacoords.y1 - bb_datacoords.y0) + 0.2
      x_position = 0

    legend_handles = []
    for color, class_name in zip(color_list, class_names):
      color_patch = mpatches.Patch(color=color, label=class_name, alpha = 0.5)
      legend_handles.append(color_patch)
    plt.legend(handles=legend_handles, fontsize = 20, bbox_to_anchor=(1.1, 0.5))
    

def clean_probs(probs):
  width = 5
  half = int(width/2)
  padded = np.pad(probs, [(half, half), (0,0)], mode = 'edge')
  weights = np.zeros([probs.shape[0], width])
  cleaned_probs = np.zeros(probs.shape)
  for i in range(probs.shape[0]):
    for j in range(width):
      sim = np.sum(np.abs(padded[i+j] - padded[i + half])) #Total variation
      sim = np.exp(-4 * sim)
      # if j == half:
      #   sim = 0
      # sim = 1
      # sim = np.exp(5 * np.sum(padded[i+j] * padded[i + half]))
      weights[i, j] = sim
    total_sim = np.sum(weights[i])
    weights[i] /= np.sum(weights[i])
    cleaned_probs[i] = np.sum(weights[i][:, None] * padded[i:i+width], 0)
    # print(total_sim)
    # cleaned_probs[i] = cleaned_probs[i]**(total_sim**0.1)
    # cleaned_probs[i] = cleaned_probs[i]**1.01
    cleaned_probs[i] /= np.sum(cleaned_probs[i])
    # cleaned_probs[i, -1] += 0.003
    cleaned_probs[i] /= np.sum(cleaned_probs[i])
  return cleaned_probs

def longest_n_blocks(probs, classes, n):
  blocks = []
  total_confs = []
  block_start = -1
  for i in range(len(probs)):
    if i == 0: #don't allow CLS token to be part of it
      continue
    if block_start == -1 and np.argmax(probs[i]) in classes:
      block_start = i
      total_confs.append(np.max(probs[i]))
    elif block_start != -1 and np.argmax(probs[i]) not in classes:
      blocks.append([block_start, i])
      block_start = -1
    elif block_start != -1 and np.argmax(probs[i]) in classes:
      total_confs[-1] += np.max(probs[i])

  new_labels = np.zeros(probs.shape[0])
  if len(blocks) > 0:
    block_lengths = [block[1] - block[0] for block in blocks]
    criteria = [conf/length for conf, length in zip(total_confs, block_lengths)]
    criteria, block_lengths, blocks = zip(*sorted(zip(criteria, block_lengths, blocks), reverse = True))
    cut_blocks = blocks[:n]
    for block in cut_blocks:
      new_labels[block[0]:block[1]] = 1
    return new_labels, blocks, block_lengths
  return new_labels, [], []

def blockify_probs(probs, classes, n_blocks):
  new_probs = np.zeros_like(probs)
  for i, indices in enumerate(classes[:-1]):
    new_probs[:, i] = longest_n_blocks(probs, indices, n_blocks)[0]
    # print(longest_n_blocks(probs, indices, n_blocks))
  new_probs[:, -1] = 0.5
  return new_probs


def blockify_probs_and_remove_duplicates(probs, classes, n_blocks, token_ids, model, similarity_threshold = 0.9, offset_map = None):
  new_probs = np.zeros_like(probs)
  highlight_char_indices = [[] for i in range(len(classes) - 1)]
  for i, indices in enumerate(classes[:-1]):
    blocks, block_lengths = longest_n_blocks(probs, indices, n_blocks)[1:]
    #turn the block lengths into token strings
    if len(blocks) > 0:
      tokens, attention_masks = block_spans_to_token_strings(blocks, block_lengths, token_ids)
      inputs = {
        'input_ids': tokens,
        'attention_mask': attention_masks
      }
      sims = get_similarities_sbert(inputs, model, None, as_tokens = True).detach().cpu().numpy()

      selected_indices = [0]
      new_probs[blocks[0][0]:blocks[0][1], i] = 1
      if offset_map is not None:
        highlight_char_indices[i].append([offset_map[blocks[0][0]][0], offset_map[blocks[0][1]-1][1]])

      last_tested = 0
      for j in range(1, len(blocks)):
        if len(selected_indices) == n_blocks:
          break
        elif np.all(sims[j, selected_indices] < similarity_threshold) :
          selected_indices.append(j)
          new_probs[blocks[j][0]:blocks[j][1], i] = 1

          if offset_map is not None:
            highlight_char_indices[i].append([offset_map[blocks[j][0]][0], offset_map[blocks[j][1]-1][1]])
    
  new_probs[:, -1] = 0.5
  if offset_map is not None:
    return highlight_char_indices
  return new_probs

def block_spans_to_token_strings(blocks, block_lengths, token_ids):
  ids = np.zeros([len(blocks), np.max(block_lengths) + 1], dtype = np.int32)
  attention_mask = np.zeros([len(blocks), np.max(block_lengths)+1], dtype = np.int32)

  ids[:, 0] = 2 #add the CLS token to the beginning
  for i, block in enumerate(blocks):
    ids[i, 1:block_lengths[i]] = token_ids[block[0]:block[1]-1]
    ids[i, block_lengths[i]] = 3 #add SEP token to the end
    attention_mask[i, 0:block_lengths[i] + 1] = 1

  return torch.from_numpy(ids).long(), torch.from_numpy(attention_mask).long()