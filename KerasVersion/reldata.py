#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, configparser, shutil, glob, itertools
from cassis import *
import numpy as np

type_system_path = './TypeSystem.xml'

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
    partition='train'):
    """"Xml ref and out dirs would typically be given for a test set"""

    self.xmi_dir = xmi_dir
    self.partition = partition

  def event_time_relations(self):
    """Make x, y etc. for a specified partition"""

    texts = []
    labels = []

    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)

    for xmi_path in glob.glob(self.xmi_dir + '*.xmi'):

      # does this xmi belong to the sought partition?
      xmi_file_name = xmi_path.split('/')[-1]
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

            texts.append(context)
            labels.append(label2int[label])

    return texts, np.array(labels)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dtr_data = RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev')

  inputs, labels = dtr_data.event_time_relations()

  print('inputs:\n', inputs[:2])
  print('labels:\n', labels[:5])
