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

label2int = {'BEFORE':0, 'OVERLAP':1, 'BEFORE/OVERLAP':2, 'AFTER':3}
int2label = {0:'BEFORE', 1:'OVERLAP', 2:'BEFORE/OVERLAP', 3:'AFTER'}

# ctakes type system types
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class DTRData:
  """Make x and y from XMI files for train, dev, or test set"""

  def __init__(
    self,
    xmi_dir,
    partition='train',
    n_files='all'):
    """Constructor"""

    self.xmi_dir = xmi_dir
    self.partition = partition
    self.n_files = None if n_files == 'all' else int(n_files)

  def read(self):
    """Make x, y etc."""

    texts = []
    labels = []

    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)

    xmi_paths = glob.glob(self.xmi_dir + '*.xmi')[:self.n_files]
    caption = 'reading %s data' % self.partition
    for xmi_path in tqdm(xmi_paths, desc=caption):

      # does this xmi belong to train, dev, or test?
      xmi_file_name = xmi_path.split('/')[-1]
      id = int(xmi_file_name.split('_')[0][-3:])
      if id % 8 not in splits[self.partition]:
        continue

      xmi_file = open(xmi_path, 'rb')
      cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
      gold_view = cas.get_view('GoldView')
      sys_view = cas.get_view('_InitialView')

      for sentence in sys_view.select(sent_type):
        sent_text = sentence.get_covered_text()

        for event in gold_view.select_covered(event_type, sentence):
          event_text = event.get_covered_text()
          dtr_label = event.event.properties.docTimeRel

          left = sent_text[: event.begin - sentence.begin]
          right = sent_text[event.end - sentence.begin :]
          context = left + ' es ' + event_text + ' ee ' + right
          context = context.replace('\n', '')

          texts.append(context)
          labels.append(label2int[dtr_label])

    return texts, labels

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dtr_data = DTRData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=10)

  inputs, labels = dtr_data.read()

  import collections
  print('unique labels:', collections.Counter(labels))

  print('inputs:\n', '\n'.join(inputs[:25]))
  print('labels:\n', labels[:25])
