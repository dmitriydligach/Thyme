#!/usr/bin/env python3
import argparse
from transformers import T5Tokenizer

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os
from tqdm import tqdm
from cassis import *
from dataset_base import ThymeDataset

# ctakes type system types
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class Data(ThymeDataset):
  """DTR data"""

  def __init__(
   self,
   xmi_dir,
   tokenizer,
   max_input_length,
   max_output_length,
   partition,
   n_files):
    """Thyme data"""

    super(Data, self).__init__(
      xmi_dir,
      tokenizer,
      max_input_length,
      max_output_length,
      n_files)

    self.partition = partition
    self.extract_events_and_dtr()

  def extract_events_and_dtr(self):
    """Extract events and times"""

    caption = 'dtr %s data' % self.partition
    for xmi_path in tqdm(self.xmi_paths, desc=caption):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in self.splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=self.type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      # iterate over sentences extracting events and times
      for sent in sys_view.select(sent_type):

        sent_text = sent.get_covered_text().replace('\n', '')

        events = []          # gold events
        events_with_dtr = [] # events and their DTRs
        metadata = []        # (note, begin, end) tuples

        for event in gold_view.select_covered(event_type, sent):

          event_text = event.get_covered_text().replace('\n', '')
          events.append(event_text)

          dtr_label = event.event.properties.docTimeRel
          events_with_dtr.append('%s|%s' % (event_text, dtr_label))

          note_name = xmi_file_name.split('.')[0]
          metadata.append((note_name, event_text, event.begin, event.end))

        input_str = 'task: DTR; sent: %s; events: %s' % (sent_text, ', '.join(events))
        self.inputs.append(input_str)

        output_str = ', '.join(events_with_dtr)
        self.outputs.append(output_str)

        self.metadata.append(metadata)

if __name__ == "__main__":
  """My main man"""

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xmi_dir=os.path.join(base, 'Thyme/Xmi/'),
    model_dir='Model/',
    model_name='t5-small',
    max_input_length=100,
    max_output_length=100,
    partition='dev',
    n_files=3)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters:', args)

  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  data = Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)

  for index in range(len(data)):
    input_ids = data[index]['input_ids']
    output_ids = data[index]['labels']
    metadata = data[index]['metadata']
    print(tokenizer.decode(input_ids, skip_special_tokens=True))
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
    print(metadata)
    print()