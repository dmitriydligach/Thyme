#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
from transformers import BertModel, BertPreTrainedModel
import os, configparser

class BertPositionalEncoding(BertPreTrainedModel):
  """Give us access to BERT's positional encodings"""

  def __init__(self, config):
    """Constructiona"""

    super(BertPositionalEncoding, self).__init__(config)
    self.bert = BertModel(config)

  def forward(self, token_ids):
    """Get a tensor of shape (batch_size, seq_len, emb_dim)"""

    # token_ids: (batch_size, seq_len)
    # need a tensor of the same shape but
    # filled with 0, 1, 2, ... in each row

    position_ids = torch.arange(
      token_ids.size(1),
      dtype=torch.long,
      device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    # now can get position embeddings of shape: (batch_size, seq_len, 768)
    bert_posit_emb = self.bert.embeddings.position_embeddings(position_ids)

    # return self.dropout(x)
    return bert_posit_emb

def main():
  """Fine-tune bert"""

  model = BertPositionalEncoding.from_pretrained(
      'bert-base-uncased')
  token_ids = torch.rand((32, 50))
  posit_emb = model(token_ids)
  print(posit_emb.shape)

if __name__ == "__main__":

  main()
