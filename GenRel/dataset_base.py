#!/usr/bin/env python3

from torch.utils.data import Dataset

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

class ThymeDataset(Dataset):
  """Thyme data"""

  def __init__(
   self,
   tokenizer,
   max_input_length,
   max_output_length):
    """Thyme data"""

    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length

    # seq2seq i/o(s)
    self.inputs = []
    self.outputs = []

    # items we need for eval
    self.metadata = []

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.inputs) == len(self.outputs))
    return len(self.inputs)

  def __getitem__(self, index):
    """Required by pytorch"""

    input = self.tokenizer(
      self.inputs[index],
      max_length=self.max_input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    output = self.tokenizer(
      self.outputs[index],
      max_length=self.max_output_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    input_ids = input.input_ids.squeeze()
    attention_mask = input.attention_mask.squeeze()
    label = output.input_ids.squeeze()

    return dict(
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels=label[1],
      metadata=self.metadata[index])

if __name__ == "__main__":
  """My main man"""

  pass