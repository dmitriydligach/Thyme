#!/usr/bin/env python3

from transformers import T5Tokenizer

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, argparse
from tqdm import tqdm
from cassis import *
from dataset_base import ThymeDataset

# ctakes type system types
rel_type = 'org.apache.ctakes.typesystem.type.relation.TemporalTextRelation'
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class Data(ThymeDataset):
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

    super(Data, self).__init__(
      xmi_dir,
      tokenizer,
      max_input_length,
      max_output_length,
      n_files)

    self.partition = partition
    self.event_time_relations()

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

  def event_time_relations(self):
    """Extract event and time relations"""

    caption = 'event-time relations in %s' % self.partition
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

      rel_lookup = Data.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')

        # extract gold events snad times
        events = []
        for event in gold_view.select_covered(event_type, sent):
          event_text = event.get_covered_text().replace('\n', '')
          events.append(event_text)

        times = []
        for time in gold_view.select_covered(time_type, sent):
          time_text = time.get_covered_text().replace('\n', '')
          times.append(time_text)

        # input string
        input_str = 'task: REL; sent: %s; events: %s; times: %s' % \
                    (sent_text, ', '.join(events), ', '.join(times))
        self.inputs.append(input_str)

        rels_in_sent = []
        # now extract event-time relations
        for event in gold_view.select_covered(event_type, sent):
          for time in gold_view.select_covered(time_type, sent):

            time_text = time.get_covered_text()
            event_text = event.get_covered_text()

            if (time, event) in rel_lookup:
              label = rel_lookup[(time, event)]
              rel_string = '%s(%s, %s)' % (label, time_text, event_text)
              rels_in_sent.append(rel_string)

            elif (event, time) in rel_lookup:
              label = rel_lookup[(event, time)]
              rel_string = '%s(%s, %s)' % (label, event_text, time_text)
              rels_in_sent.append(rel_string)

        if len(rels_in_sent) == 0:
          self.outputs.append('no event-time relations')
        else:
          self.outputs.append(' '.join(rels_in_sent))

  def events_event_relations(self):
    """Very eventful"""

    caption = 'event-event relations in %s' % self.partition
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

      rel_lookup = Data.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('Relation extraction: ' + sent_text)

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
      if id % 8 not in self.splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=self.type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = Data.index_relations(gold_view)

      # iterate over sentences extracting relations
      for sent in sys_view.select(sent_type):
        sent_text = sent.get_covered_text().replace('\n', '')
        self.inputs.append('Relation extraction: ' + sent_text)

        # event-time relations in this sentence
        et_rels_in_sent = []

        for event in gold_view.select_covered(event_type, sent):
          for time in gold_view.select_covered(time_type, sent):

            if (time, event) in rel_lookup:
              label = rel_lookup[(time, event)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, time_text, event_text)
              et_rels_in_sent.append(rel_string)

            if (event, time) in rel_lookup:
              label = rel_lookup[(event, time)]
              time_text = time.get_covered_text()
              event_text = event.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event_text, time_text)
              et_rels_in_sent.append(rel_string)

        et_output = 'event-time relations: '
        if len(et_rels_in_sent) == 0:
          et_output = et_output + 'none'
        else:
          et_output = et_output + ' '.join(et_rels_in_sent)

        # event-event relations in this sentence
        ee_rels_in_sent = []

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
              ee_rels_in_sent.append(rel_string)

            if (event2, event1) in rel_lookup:
              label = rel_lookup[(event2, event1)]
              event1_text = event1.get_covered_text()
              event2_text = event2.get_covered_text()
              rel_string = '%s(%s, %s)' % (label, event2_text, event1_text)
              ee_rels_in_sent.append(rel_string)

        ee_output = 'event-event relations: '
        if len(ee_rels_in_sent) == 0:
          ee_output = ee_output + 'none'
        else:
          ee_output = ee_output + ' '.join(ee_rels_in_sent)

        self.outputs.append(et_output + '; ' + ee_output)

def main():
  """This is where it happens"""

  tok = T5Tokenizer.from_pretrained('t5-small')
  data = Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tok,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)

  for index in range(len(data)):
    input_ids = data[index]['input_ids']
    output_ids = data[index]['labels']
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
    max_input_length=200,
    max_output_length=200,
    partition='dev',
    n_files=5)

  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  main()
