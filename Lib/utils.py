#!/usr/bin/env python3

import torch
import numpy as np
from transformers import BertTokenizer

def to_inputs(texts, pad_token=0):
  """Converts texts into input matrices required by BERT"""

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
  shape = (len(rows), max(len(row) for row in rows))
  token_ids = np.full(shape=shape, fill_value=pad_token)
  is_token = np.zeros(shape=shape)

  for i, row in enumerate(rows):
    token_ids[i, :len(row)] = row
    is_token[i, :len(row)] = 1

  token_ids = torch.tensor(token_ids)
  is_token = torch.tensor(is_token)

  return token_ids, is_token

if __name__ == "__main__":

  print()
