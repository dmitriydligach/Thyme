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

type_system_path = './TypeSystem.xml'

splits = {
  'train': set([0,1,2,3]),
  'dev': set([4,5]),
  'test': set([6,7])}

label2int = {'NONE':0, 'CONTAINS':1, 'CONTAINS-1':2}
int2label = {0:'NONE', 1:'CONTAINS', 2:'CONTAINS-1'}

# ctakes type system types
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class Thyme(Dataset):
  """Thyme data"""

  def __init__(
   self,
   xmi_dir,
   tokenizer,
   input_length,
   output_length,
   partition='train',
   n_files='all'):
    """Thyme data"""

    self.xmi_dir = xmi_dir
    self.tokenizer = tokenizer
    self.input_length = input_length
    self.output_length = output_length
    self.partition = partition
    self.n_files = None if n_files == 'all' else int(n_files)

  def events_and_times(self):
    """Extract events and times"""

    inputs = []  # source text
    outputs = [] # events and times

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
        inputs.append('IE: ' + sent_text)

        # events and times for now
        output = []

        output.append('events: ')
        for event in gold_view.select_covered(event_type, sent):
          event_text = event.get_covered_text().replace('\n', '')
          output.append(event_text)

        output.append('times: ')
        for time in gold_view.select_covered(time_type, sent):
          time_text = time.get_covered_text().replace('\n', '')
          output.append(time_text)

        outputs.append(' '.join(output))

    return inputs, outputs

  def __len__(self):
    """Requried by pytorch"""

    return self.dataset.shape[0]

  def clean_text(self, text):
    """Do we even need this?"""

    text = text.replace('\n', '')
    text = text.replace('``', '')
    text = text.replace('"', '')

    return text

  def to_int_seqs(self, instance):
    """Prepare inputs and outputs"""

    text = self.clean_text(instance['text'])
    summary = self.clean_text(instance['headline'])

    text = self.tokenizer(
      text,
      max_length=self.input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    summary = self.tokenizer(
      summary,
      max_length=self.output_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return text, summary

  def __getitem__(self, index):
    """Required by pytorch"""

    text, summary = self.to_int_seqs(self.dataset[index])

    input_ids = text.input_ids.squeeze()
    input_mask = text.attention_mask.squeeze()

    output_ids = summary.input_ids.squeeze()
    output_mask = summary.attention_mask.squeeze()

    return input_ids, input_mask, output_ids, output_mask

if __name__ == "__main__":
  """My main man"""

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    model_name='t5-small',
    max_input_length=50,
    max_output_length=50,
    partition='train',
    n_files=10)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters:', args)

  data = Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=T5Tokenizer.from_pretrained('t5-small'),
    input_length=args.max_input_length,
    output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)
  inputs, outputs = data.events_and_times()

  for input, output in zip(inputs, outputs):
    print(input)
    print(output)
    print()