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

event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class DTRData:
  """Make x and y from XMI files for train, dev, or test set"""

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

      # does this xmi belong to train, dev, or test?
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

          inputs.append(tokenizer.encode(context))
          labels.append(label2int[dtr_label])

          note_name = xmi_file_name.split('.')[0]
          self.offsets.append((note_name, event.begin, event.end))

    inputs = pad_sequences(
      inputs,
      maxlen=128, # TODO: what is this???
      dtype='long',
      truncating='post',
      padding='post')

    masks = [] # attention masks
    for sequence in inputs:
      mask = [float(value > 0) for value in sequence]
      masks.append(mask)

    return inputs, labels, masks

  def write(self, predictions):
    """Write predictions in anafora XML format"""

    # predictions are in the same order in which they were read
    prediction_lookup = dict(zip(self.offsets, predictions))

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
        if (sub_dir, start, end) not in prediction_lookup:
          print('missing key:', (sub_dir, start, end))
          continue

        label = prediction_lookup[(sub_dir, start, end)]
        entity.properties['DocTimeRel'] = int2label[label]

        data.annotations.append(entity)

      data.indent()
      os.mkdir(os.path.join(self.xml_out_dir, sub_dir))
      out_path = os.path.join(self.xml_out_dir, sub_dir, file_names[0])
      data.to_file(out_path)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dtr_data = DTRData(
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

  # predictions = [label for label in labels]
  # dtr_data.write(predictions)
