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

label2int = {'NONE':0, 'CONTAINS':1, 'CONTAINS-1':2}
int2label = {0:'NONE', 1:'CONTAINS', 2:'CONTAINS-1'}

# ctakes type system types
rel_type = 'org.apache.ctakes.typesystem.type.relation.TemporalTextRelation'
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'
token_type = 'org.apache.ctakes.typesystem.type.syntax.WordToken'

# TODO: generating about 1/2 of the relations we need to generate?
# TODO: use base token instead of word token to get punctuation

def index_relations(gold_view):
  """Map arguments to relation types"""

  rel_lookup = {}

  for rel in gold_view.select(rel_type):
    arg1 = rel.arg1.argument
    arg2 = rel.arg2.argument

    if rel.category == 'CONTAINS':
      rel_lookup[(arg1, arg2)] = rel.category

  return rel_lookup

def get_context(sys_view, sent, larg, rarg, lmarker, rmarker):
  """Build a context string using left and right arguments and their markers"""

  sent_text = sent.get_covered_text()
  left_text = larg.get_covered_text()
  right_text = rarg.get_covered_text()

  left_context = sent_text[: larg.begin - sent.begin]
  middle_context = sent_text[larg.end - sent.begin : rarg.begin - sent.begin]
  right_context = sent_text[rarg.end - sent.begin :]

  left_start = ' [s%s] ' % lmarker
  left_end = ' [e%s] ' % lmarker
  right_start = ' [s%s] ' % rmarker
  right_end = ' [e%s] ' % rmarker

  context = left_context + left_start + left_text + left_end + \
            middle_context + right_start + right_text + \
            right_end + right_context

  return context.replace('\n', '')

class RelData:
  """Make x and y from XMI files for train, dev, or test partition"""

  def __init__(
    self,
    xmi_dir,
    partition='train',
    xml_ref_dir=None,
    xml_out_dir=None):
    """Xml ref and out dirs would typically be given for a test set"""

    self.xmi_dir = xmi_dir
    self.partition = partition
    self.xml_ref_dir = xml_ref_dir
    self.xml_out_dir = xml_out_dir

    # (note_id, begin, end) tuples
    self.offsets = []

  def event_time_relations(self):
    """Make x, y etc. for a specified partition"""

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

      # does this xmi belong to the sought partition?
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = index_relations(gold_view)

      # iterate over sentences, extracting relations
      for sent in sys_view.select(sent_type):

        for event in gold_view.select_covered(event_type, sent):
          for time in gold_view.select_covered(time_type, sent):

            label = 'NONE'
            if (time, event) in rel_lookup:
              label = rel_lookup[(time, event)]
            if (event, time) in rel_lookup:
              label = rel_lookup[(event, time)] + '-1'

            if time.begin < event.begin:
              context = get_context(sys_view, sent, time, event, 't', 'e')
            else:
              context = get_context(sys_view, sent, event, time, 'e', 't')

            inputs.append(tokenizer.encode(context))
            labels.append(label2int[label])

            # print('%s|%s' % (label, context))
            # note_name = xmi_file_name.split('.')[0]
            # self.offsets.append((note_name, event.begin, event.end))

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

  def event_event_relations(self):
    """Make x, y etc. for a specified partition"""

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

      # does this xmi belong to the sought partition?
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      rel_lookup = index_relations(gold_view)

      # iterate over sentences, extracting relations
      for sent in sys_view.select(sent_type):

        for event1 in gold_view.select_covered(event_type, sent):
          for event2 in gold_view.select_covered(event_type, sent):

            label = 'NONE'
            if (event1, event2) in rel_lookup:
              label = rel_lookup[(event1, event2)]
            if (event2, event1) in rel_lookup:
              label = rel_lookup[(event2, event1)] + '-1'

            if event1.begin < event2.begin:
              context = get_context(sys_view, sent, event1, event2, 'e1', 'e2')
            else:
              context = get_context(sys_view, sent, event2, event1, 'e2', 'e1')

            if label != 'NONE': print(label + "|" + context)

            inputs.append(tokenizer.encode(context))
            labels.append(label2int[label])

            # print('%s|%s' % (label, context))
            # note_name = xmi_file_name.split('.')[0]
            # self.offsets.append((note_name, event.begin, event.end))

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

  inputs, labels, masks = dtr_data.event_event_relations()

  print('inputs:\n', inputs[:2])
  print('labels:\n', labels[:5])
  print('masks:\n', masks[:1])

  print('offsets:\n', dtr_data.offsets[:50])

  print('inputs shape:', inputs.shape)
  print('number of labels:', len(labels))
  print('number of masks:', len(masks))
