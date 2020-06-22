#!/usr/bin/env python3

import torch
import numpy as np
from transformers import BertTokenizer
from tokenizers import CharBPETokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

def to_bert_inputs(texts, max_len=None, pad_token=0):
  """Converts texts into input matrices required by BERT"""

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  id_seqs = [tokenizer.encode(text, add_special_tokens=True) for text in texts]

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in id_seqs)
  shape = (len(id_seqs), max_len)

  token_ids = np.full(shape=shape, fill_value=pad_token)
  is_token = np.zeros(shape=shape)

  for i, id_seq in enumerate(id_seqs):
    if len(id_seq) > max_len:
      id_seq = id_seq[:max_len]
    token_ids[i, :len(id_seq)] = id_seq
    is_token[i, :len(id_seq)] = 1

  token_ids = torch.tensor(token_ids)
  is_token = torch.tensor(is_token)

  return token_ids, is_token

def to_transformer_inputs(texts, max_len=None):
  """Matrix of token ids and a square attention mask for eash sample"""

  tokenizer = CharBPETokenizer(
    '../Tokenize/thyme-tokenizer-vocab.json',
    '../Tokenize/thyme-tokenizer-merges.txt')
  seqs = [tokenizer.encode(text).ids for text in texts]

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
  mask = torch.zeros(len(seqs), max_len, max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    ids[i, :len(seq)] = torch.tensor(seq)
    mask[i, :len(seq), :len(seq)] = 1

  return ids, mask

def to_token_id_sequences(texts, max_len=None):
  """Matrix of token ids"""

  tokenizer = CharBPETokenizer(
    '../Tokenize/thyme-tokenizer-vocab.json',
    '../Tokenize/thyme-tokenizer-merges.txt')
  seqs = [tokenizer.encode(text).ids for text in texts]

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  ids = torch.zeros(len(seqs), max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    ids[i, :len(seq)] = torch.tensor(seq)

  return ids

def to_lstm_inputs(texts, max_len=None):
  """Padded at the beginning rather than at the end"""

  tokenizer = CharBPETokenizer(
    '../Tokenize/thyme-tokenizer-vocab.json',
    '../Tokenize/thyme-tokenizer-merges.txt')
  seqs = [tokenizer.encode(text).ids for text in texts]

  if max_len is None:
    # set max_len to the length of the longest sequence
    max_len = max(len(id_seq) for id_seq in seqs)

  ids = torch.zeros(len(seqs), max_len, dtype=torch.long)

  for i, seq in enumerate(seqs):
    if len(seq) > max_len:
      seq = seq[:max_len]
    ids[i, -len(seq):] = torch.tensor(seq)

  return ids

def make_data_loader(texts, labels, batch_size, max_len, partition, input_processor):
  """DataLoader objects for train or dev/test sets"""

  model_inputs = input_processor(texts, max_len)
  labels = torch.tensor(labels)

  # e.g. transformers take input ids and attn masks
  if type(model_inputs) is tuple:
    tensor_dataset = TensorDataset(*model_inputs, labels)
  else:
    tensor_dataset = TensorDataset(model_inputs, labels)

  # use sequential sampler for dev and test
  if partition == 'train':
    sampler = RandomSampler(tensor_dataset)
  else:
    sampler = SequentialSampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=sampler,
    batch_size=batch_size)

  return data_loader

if __name__ == "__main__":

  texts = ['it is happening again',
           'the owls are not what they seem']
  ids, masks = to_transformer_inputs(texts, max_len=None)
  print('ids:', ids)
  print('masks:', masks)