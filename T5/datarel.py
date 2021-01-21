#!/usr/bin/env python3

from torch.utils.data import Dataset
from transformers import T5Tokenizer

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, glob, argparse
from tqdm import tqdm
from cassis import *

splits = {
  'train': set([0,1,2,3]),
  'dev': set([4,5]),
  'test': set([6,7])}

# ctakes type system types
rel_type = 'org.apache.ctakes.typesystem.type.relation.TemporalTextRelation'
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
   n_files):
    """Thyme data"""

    self.xmi_dir = xmi_dir
    self.tokenizer = tokenizer
    self.max_input_length = max_input_length
    self.max_output_length = max_output_length
    self.partition = partition
    self.n_files = None if n_files == 'all' else int(n_files)

    type_system_file = open(type_system_path, 'rb')
    self.type_system = load_typesystem(type_system_file)
    self.xmi_paths = glob.glob(self.xmi_dir + '*.xmi')[:self.n_files]

    self.inputs = []
    self.outputs = []

    # self.extract_events_time_relations()
    # self.extract_events_event_relations()
    self.extract_all_relations()

  @staticmethod
  def index_relations(gold_view):
    """Map arguments to relation types"""

    rel_lookup = {}
    for rel in gold_view.select(rel_type):
      arg1 = rel.arg1.argument
      arg2 = rel.arg2.argument
      if rel.category == 'CONTAINS':
        rel_lookup[(arg1, arg2)] = rel.category

    return rel_lookup

  def extract_events_time_relations(self):
    """Extract events and times"""

    caption = 'event-time relations in %s' % self.partition
    for xmi_path in tqdm(self.xmi_paths, desc=caption):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=self.type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = Thyme.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('Perform IE: ' + sent_text)

        rels_in_sent = []
        for event in gold_view.select_covered(event_type, sent):
          for time in gold_view.select_covered(time_type, sent):

            if (time, event) in rel_lookup:
              label = rel_lookup[(time, event)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, time_text, event_text)
              rels_in_sent.append(rel_string)

            if (event, time) in rel_lookup:
              label = rel_lookup[(event, time)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event_text, time_text)
              rels_in_sent.append(rel_string)

        if len(rels_in_sent) == 0:
          self.outputs.append('no event-time relations')
        else:
          self.outputs.append('event-time rels: ' + ' '.join(rels_in_sent))

  def extract_events_event_relations(self):
    """Very eventful"""

    caption = 'event-event relations in %s' % self.partition
    for xmi_path in tqdm(self.xmi_paths, desc=caption):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=self.type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = Thyme.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('Perform IE: ' + sent_text)

        rels_in_sent = []
        events_in_sent = list(gold_view.select_covered(event_type, sent))
        for i in range(0, len(events_in_sent)):
          for j in range(i + 1,  len(events_in_sent)):

            event1 = events_in_sent[i]
            event2 = events_in_sent[j]

            if (event1, event2) in rel_lookup:
              label = rel_lookup[(event1, event2)]
              event1_text = event1.get_covered_text()
              event2_text = event2.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event1_text, event2_text)
              rels_in_sent.append(rel_string)

            if (event2, event1) in rel_lookup:
              label = rel_lookup[(event2, event1)]
              event1_text = event1.get_covered_text()
              event2_text = event2.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event2_text, event1_text)
              rels_in_sent.append(rel_string)

        if len(rels_in_sent) == 0:
          self.outputs.append('no event-event relations')
        else:
          self.outputs.append('event-event rels: ' + ' '.join(rels_in_sent))

  def extract_all_relations(self):
    """Extract ee and et relations"""

    caption = 'all relations in %s' % self.partition
    for xmi_path in tqdm(self.xmi_paths, desc=caption):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=self.type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = Thyme.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('Perform IE: ' + sent_text)

        rels_in_sent = []

        # get event-time relations first
        rels_in_sent.append('event-time relations:')
        for event in gold_view.select_covered(event_type, sent):
          for time in gold_view.select_covered(time_type, sent):

            if (time, event) in rel_lookup:
              label = rel_lookup[(time, event)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, time_text, event_text)
              rels_in_sent.append(rel_string)

            if (event, time) in rel_lookup:
              label = rel_lookup[(event, time)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event_text, time_text)
              rels_in_sent.append(rel_string)

        # now get event-event relations
        rels_in_sent.append('event-event relations:')
        events_in_sent = list(gold_view.select_covered(event_type, sent))
        for i in range(0, len(events_in_sent)):
          for j in range(i + 1,  len(events_in_sent)):

            event1 = events_in_sent[i]
            event2 = events_in_sent[j]

            if (event1, event2) in rel_lookup:
              label = rel_lookup[(event1, event2)]
              event1_text = event1.get_covered_text()
              event2_text = event2.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event1_text, event2_text)
              rels_in_sent.append(rel_string)

            if (event2, event1) in rel_lookup:
              label = rel_lookup[(event2, event1)]
              event1_text = event1.get_covered_text()
              event2_text = event2.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event2_text, event1_text)
              rels_in_sent.append(rel_string)

        if len(rels_in_sent) == 0:
          self.outputs.append('no relations in this sentence')
        else:
          self.outputs.append(' '.join(rels_in_sent))

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

def main():
  """This is where it happens"""

  tok = T5Tokenizer.from_pretrained('t5-small')
  data = Thyme(
    xmi_dir=args.xmi_dir,
    tokenizer=tok,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)

  for index in range(len(data)):
    input_ids, input_mask, output_ids, output_mask = data[index]
    print(tok.decode(input_ids, skip_special_tokens=True))
    print(tok.decode(output_ids, skip_special_tokens=True))
    print()

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
    n_files=5)

  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()
