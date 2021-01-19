#!/usr/bin/env python3
import argparse

from torch.utils.data import Dataset
from transformers import T5Tokenizer

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, glob
from tqdm import tqdm
from cassis import *

splits = {
  'train': set([0,1,2,3]),
  'dev': set([4,5]),
  'test': set([6,7])}

# ctakes type system types
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

# ctakes type system
type_system_path='./TypeSystem.xml'

class Thyme(Dataset):
  """Thyme data"""

  def __init__(
   self,
   xmi_dir,
   tokenizer,
   max_input_length,
   max_output_length,
   partition,
   n_files='all'):
    """Thyme data"""

    self.xmi_dir = xmi_dir
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length
    self.partition = partition
    self.n_files = None if n_files == 'all' else int(n_files)

    self.inputs = []
    self.outputs = []
    self.extract_events_and_dtr()

  def extract_events_and_dtr(self):
    """Extract events and times"""

    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)

    xmi_paths = glob.glob(self.xmi_dir + '*.xmi')[:self.n_files]
    caption = 'reading %s data' % self.partition

    for xmi_path in tqdm(xmi_paths, desc=caption):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      # iterate over sentences extracting events and times
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('perform IE: ' + sent_text)

        events = [] # events and their DTRs
        for event in gold_view.select_covered(event_type, sent):
          event_text = event.get_covered_text().replace('\n', '')
          dtr_label = event.event.properties.docTimeRel
          events.append('%s/%s' % (event_text, dtr_label))

        output_string = 'events: ' + ', '.join(events)
        self.outputs.append(output_string)

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
    input_mask = input.attention_mask.squeeze()

    output_ids = output.input_ids.squeeze()
    output_mask = output.attention_mask.squeeze()

    return input_ids, input_mask, output_ids, output_mask

if __name__ == "__main__":
  """My main man"""

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    model_dir='Model/',
    model_name='t5-small',
    max_input_length=50,
    max_output_length=50,
    partition='dev',
    n_files=3)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters:', args)

  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  data = Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)

  for index in range(len(data)):
    input_ids, input_mask, output_ids, output_mask = data[index]
    print(tokenizer.decode(input_ids, skip_special_tokens=True))
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
    print()