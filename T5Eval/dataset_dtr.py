#!/usr/bin/env python3

import argparse
from transformers import T5Tokenizer

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, shutil
from tqdm import tqdm
from cassis import *
import anafora
from dataset_base import ThymeDataset
from torch.utils.data import DataLoader

# ctakes type system types
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

# files containing event annotations
xml_regex = '.*[.]Temporal.*[.]xml'

class Data(ThymeDataset):
  """DTR data"""

  def __init__(
   self,
   xmi_dir,
   xml_ref_dir,
   xml_out_dir,
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
    self.xml_ref_dir = xml_ref_dir
    self.xml_out_dir = xml_out_dir

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
          metadata_tuple = (note_name, str(event.begin), str(event.end))
          metadata.append('|'.join(metadata_tuple))

        input_str = 'task: DTR; sent: %s; events: %s' % (sent_text, ', '.join(events))
        self.inputs.append(input_str)

        output_str = ', '.join(events_with_dtr)
        self.outputs.append(output_str)

        metadata_str = '||'.join(metadata)
        self.metadata.append(metadata_str)

  def write_xml(self, prediction_lookup):
    """Write predictions in anafora XML format"""

    # make a directory to write anafora xml
    if os.path.isdir(self.xml_out_dir):
      shutil.rmtree(self.xml_out_dir)
    os.mkdir(self.xml_out_dir)

    # iterate over reference xml files
    # look up the DTR prediction for each event
    # and write it in anafora format to specificed dir
    for sub_dir, text_name, file_names in \
            anafora.walk(self.xml_ref_dir, xml_regex):

      path = os.path.join(self.xml_ref_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(path)
      data = anafora.AnaforaData()

      for event in ref_data.annotations.select_type('EVENT'):

        # make a new entity and copy some ref info
        entity = anafora.AnaforaEntity()
        entity.id = event.id
        start, end = event.spans[0]
        entity.spans = event.spans
        entity.type = event.type

        # lookup the prediction
        key = '|'.join((sub_dir, str(start), str(end)))
        if key not in prediction_lookup:
          print('missing key:', key)
          continue

        entity.properties['DocTimeRel'] = prediction_lookup[key]
        data.annotations.append(entity)

      data.indent()
      os.mkdir(os.path.join(self.xml_out_dir, sub_dir))
      out_path = os.path.join(self.xml_out_dir, sub_dir, file_names[0])
      data.to_file(out_path)

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
    n_files=25)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters:', args)

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)
  dataset = Data(
    xmi_dir=args.xmi_dir,
    tokenizer=tokenizer,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    partition=args.partition,
    n_files=args.n_files)
  data_loader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=16)

  for instance in dataset:
    print('[input]', tokenizer.decode(instance['input_ids'], skip_special_tokens=True))
    print('[output]', tokenizer.decode(instance['labels'], skip_special_tokens=True))
    print('[metadata]', instance['metadata'], '\n')

  # for batch in data_loader:
  #   if(len(batch['metadata'])) > 0:
  #     print(len(batch['metadata'][0]))
  #   print(batch['metadata'])