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
   max_input_length):
    """Thyme data"""

    self.tokenizer = tokenizer
    self.max_input_length = max_input_length

    # input seqs and their labels
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

    # call PreTrainedTokenizerBase.__call__()
    input = self.tokenizer(
      self.inputs[index],
      max_length=self.max_input_length,
      add_special_tokens=True,
      return_token_type_ids=False,
      return_attention_mask=True,
      padding='max_length',
      truncation=True,
      return_tensors='pt',
      verbose=True)

    label = self.outputs[index]
    if label == '_':
      label = 99

    return dict(
      input_ids=input.input_ids.squeeze(),
      attention_mask=input.attention_mask.squeeze(),
      # labels=output.input_ids[0],
      labels=int(label),
      metadata=self.metadata[index])

if __name__ == "__main__":
  """My main man"""

  pass