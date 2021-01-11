#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, configparser, glob
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
rel_type = 'org.apache.ctakes.typesystem.type.relation.TemporalTextRelation'
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
time_type = 'org.apache.ctakes.typesystem.type.textsem.TimeMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

def index_relations(gold_view):
  """Map arguments to relation types"""

  rel_lookup = {}

  for rel in gold_view.select(rel_type):
    arg1 = rel.arg1.argument
    arg2 = rel.arg2.argument

    if rel.category == 'CONTAINS':
      rel_lookup[(arg1, arg2)] = rel.category

  return rel_lookup

def get_context(sent, larg, rarg, lmarker, rmarker):
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
    n_files='all'):
    """"Xml ref and out dirs would typically be given for a test set"""

    self.xmi_dir = xmi_dir
    self.partition = partition
    self.n_files = None if n_files == 'all' else int(n_files)

  def event_time_relations(self):
    """Make x, y etc. for a specified partition"""

    texts = []
    labels = []

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
              context = get_context(sent, time, event, 't', 'e')
            else:
              context = get_context(sent, event, time, 'e', 't')

            texts.append(context)
            labels.append(label2int[label])

    return texts, labels

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  rel_data = RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=10)

  inputs, labels = rel_data.event_time_relations()

  import collections
  print('unique labels:', collections.Counter(labels))

  # print a few 'contains' samples
  for input, label in zip(inputs, labels):
    if label == 1:
      print(input)
