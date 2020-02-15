#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os, configparser, shutil, glob
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from cassis import *
import anafora

label2int = {'BEFORE':0, 'OVERLAP':1, 'BEFORE/OVERLAP':2, 'AFTER':3}
int2label = {0:'BEFORE', 1:'OVERLAP', 2:'BEFORE/OVERLAP', 3:'AFTER'}

event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

class DTRData:
  """Make x and y from raw data"""

  def __init__(self, xmi_dir, out_dir, max_length):
    """Constructor"""

    self.xmi_dir = xmi_dir
    self.out_dir = out_dir
    self.max_length = max_length

  def read(self):
    """Make x, y etc."""

    inputs = []
    labels = []

    tokenizer = BertTokenizer.from_pretrained(
      'bert-base-uncased',
      do_lower_case=True)

    type_system_file = open(cfg.get('data', 'type_system'), 'rb')
    type_system = load_typesystem(type_system_file)

    for xmi_path in glob.glob(self.xmi_dir + '*.xmi'):
      print(xmi_path)

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

          inputs.append(tokenizer.encode(context.replace('\n', '')))
          labels.append(label2int[dtr_label])

    inputs = pad_sequences(
      inputs,
      maxlen=self.max_length,
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

    index = 0

    if os.path.isdir(self.out_dir):
      shutil.rmtree(self.out_dir)
    os.mkdir(self.out_dir)

    for sub_dir, text_name, file_names in \
            anafora.walk(self.xml_dir, self.xml_regex):

      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      data = anafora.AnaforaData()

      for event in ref_data.annotations.select_type('EVENT'):
        entity = anafora.AnaforaEntity()

        entity.id = event.id
        start, end = event.spans[0]
        entity.spans = event.spans
        entity.type = event.type
        entity.properties['DocTimeRel'] = int2label[predictions[index]]

        data.annotations.append(entity)
        index = index + 1

      os.mkdir(os.path.join(self.out_dir, sub_dir))
      out_path = os.path.join(self.out_dir, sub_dir, file_names[0])

      data.indent()
      data.to_file(out_path)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dtr_data = DTRData(
    os.path.join(base, cfg.get('data', 'dev_xmi')),
    cfg.get('data', 'out_dir'),
    cfg.getint('bert', 'max_len'))
  inputs, labels, masks = dtr_data.read()

  print('inputs:\n', inputs[:1])
  print('labels:\n', labels[:5])
  print('masks:\n', masks[:1])

  print('inputs shape:', inputs.shape)
  print('number of labels:', len(labels))
  print('number of masks:', len(masks))

  predictions = [label for label in labels]
  dtr_data.write(predictions)
