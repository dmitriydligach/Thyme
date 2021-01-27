#!/usr/bin/env python3

from torch.utils.data import Dataset

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import glob
from cassis import *

# ctakes type system
type_system_path='./TypeSystem.xml'

class ThymeDataset(Dataset):
  """Thyme data"""

  def __init__(
   self,
   xmi_dir,
   tokenizer,
   max_input_length,
   max_output_length,
   n_files):
    """Thyme data"""

    self.xmi_dir = xmi_dir
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length
    self.n_files = None if n_files == 'all' else int(n_files)

    # get type system to read xmi files
    type_system_file = open(type_system_path, 'rb')
    self.type_system = load_typesystem(type_system_file)
    self.xmi_paths = glob.glob(self.xmi_dir + '*.xmi')[:self.n_files]

    # seq2seq i/o(s)
    self.inputs = []
    self.outputs = []

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
    decoder_input_ids = output.input_ids.squeeze()
    decoder_attention_mask = output.attention_mask.squeeze()

    return dict(
      input_ids=input_ids,
      attention_mask=attention_mask,
      decoder_input_ids=decoder_input_ids,
      decoder_attention_mask=decoder_attention_mask)

if __name__ == "__main__":
  """My main man"""

  pass