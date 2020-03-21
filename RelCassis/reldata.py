#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, configparser, shutil, glob
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from cassis import *
import anafora

type_system_path = './TypeSystem.xml'
xml_regex = '.*[.]Temporal.*[.]xml'

splits = {
  'train': set([0,1,2,3]),
  'dev': set([4,5]),
  'test': set([6,7])}

label2int = {'BEFORE':0, 'OVERLAP':1, 'BEFORE/OVERLAP':2, 'AFTER':3}
int2label = {0:'BEFORE', 1:'OVERLAP', 2:'BEFORE/OVERLAP', 3:'AFTER'}

rel_type = 'org.apache.ctakes.typesystem.type.relation.TemporalTextRelation'
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

def index_relations(gold_view):
  """Map arguments to relation types"""

  rel_lookup = {}

  for rel in gold_view.select(rel_type):
    a1 = rel.arg1.argument
    a2 = rel.arg2.argument

    if rel.category == 'CONTAINS':
      rel_lookup[(a1, a2)] = rel.category

  return rel_lookup

class RelData:
  """Make x and y from XMI files for train, dev, or test partition"""

  def __init__(
    self,
    xmi_dir,
    partition='train',
    xml_ref_dir=None,
    xml_out_dir=None):
    """Constructor"""

    self.xmi_dir = xmi_dir
    self.partition = partition
    self.xml_ref_dir = xml_ref_dir
    self.xml_out_dir = xml_out_dir

    # (note_id, begin, end) tuples
    self.offsets = []

  def read(self):
    """Make x, y etc."""

    inputs = []
    labels = []

    tokenizer = BertTokenizer.from_pretrained(
      'bert-base-uncased',
      do_lower_case=True)

    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)

    # read xmi files and make instances to feed into bert
    for xmi_path in glob.glob(self.xmi_dir + '*.xmi'):
      xmi_file_name = xmi_path.split('/')[-1]

      # does this xmi belong to the right partition?
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = index_relations(gold_view)

      # iterate over sentences, extracting relations
      for sentence in sys_view.select(sent_type):
        sent_text = sentence.get_covered_text()

        for event in gold_view.select_covered(event_type, sentence):
          for time in gold_view.select_covered(time_type, sentence):

            label = 'none'

            if (time, event) in rel_lookup:
              label = 'contains'
              print('time-event:', rel_lookup[(time, event)])

            if (event, time) in rel_lookup:
              label = 'contains-1'
              print('event-time:', rel_lookup[(event, time)])

            event_text = event.get_covered_text()
            dtr_label = event.event.properties.docTimeRel

            left = sent_text[: event.begin - sentence.begin]
            right = sent_text[event.end - sentence.begin :]
            context = left + ' es ' + event_text + ' ee ' + right
            context = context.replace('\n', '')

            inputs.append(tokenizer.encode(context))
            labels.append(label2int[dtr_label])

            note_name = xmi_file_name.split('.')[0]
            self.offsets.append((note_name, event.begin, event.end))

    inputs = pad_sequences(
      inputs,
      maxlen=max([len(seq) for seq in inputs]),
      dtype='long',
      truncating='post',
      padding='post')

    masks = [] # attention masks
    for sequence in inputs:
      mask = [float(value > 0) for value in sequence]
      masks.append(mask)

    return inputs, labels, masks

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dtr_data = RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    xml_ref_dir=os.path.join(base, cfg.get('data', 'ref_xml_dir')),
    xml_out_dir=cfg.get('data', 'out_xml_dir'))

  inputs, labels, masks = dtr_data.read()

  print('inputs:\n', inputs[:1])
  print('labels:\n', labels[:5])
  print('masks:\n', masks[:1])

  print('offsets:\n', dtr_data.offsets[:50])

  print('inputs shape:', inputs.shape)
  print('number of labels:', len(labels))
  print('number of masks:', len(masks))
