#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('./Anafora')

import os, configparser

from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

import anafora

label2int = {'BEFORE':0, 'OVERLAP':1, 'BEFORE/OVERLAP':2, 'AFTER':3}
int2label = {0:'BEFORE', 1:'OVERLAP', 2:'BEFORE/OVERLAP', 3:'AFTER'}

# TODO: All questions  es were answ ee ered.

class DTRData:
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex,
    context_chars,
    max_length):
    """Constructor"""

    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.xml_regex = xml_regex
    self.context_chars = context_chars
    self.max_length = max_length

  def __call__(self):
    """Make x, y etc."""

    inputs = []
    labels = []

    tokenizer = BertTokenizer.from_pretrained(
      'bert-base-uncased',
      do_lower_case=True)

    for sub_dir, text_name, file_names in \
            anafora.walk(self.xml_dir, self.xml_regex):
      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      text_path = os.path.join(self.text_dir, text_name)
      text = open(text_path).read()

      for event in ref_data.annotations.select_type('EVENT'):
        label = event.properties['DocTimeRel']
        labels.append(label2int[label])

        start, end = event.spans[0]
        event = text[start:end] # should be end+1?
        left = text[start - self.context_chars : start]
        right = text[end : end + self.context_chars]

        context = left + ' es ' + event + ' ee ' + right
        inputs.append(tokenizer.encode(context.replace('\n', '')))

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

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  xml_dir = os.path.join(base, cfg.get('data', 'dev_xml'))
  text_dir = os.path.join(base, cfg.get('data', 'dev_text'))
  xml_regex = cfg.get('data', 'xml_regex')
  context_chars = cfg.getint('args', 'context_chars')

  dtr_data = DTRData(
    xml_dir,
    text_dir,
    xml_regex,
    cfg.getint('args', 'context_chars'),
    cfg.getint('bert', 'max_len'))
  inputs, labels, masks = dtr_data()

  print('inputs shape:', inputs.shape)
  print('number of labels:', len(labels))
  print('number of masks:', len(masks))
