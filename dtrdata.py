#!/usr/bin/env python3

import os, sys, configparser
sys.path.append('./Anafora')

from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

import anafora

label2int = {'BEFORE':0, 'OVERLAP':1, 'BEFORE/OVERLAP':2, 'AFTER':3}

class DTRData:
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex,
    context_size,
    max_length):
    """Constructor"""

    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.xml_regex = xml_regex
    self.context_size = context_size
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
      text = open(text_path).read().replace('\n', '')

      for event in ref_data.annotations.select_type('EVENT'):
        label = event.properties['DocTimeRel']
        labels.append(label2int[label])

        start, end = event.spans[0]
        event = text[start:end]
        left = text[start - self.context_size : start]
        right = text[end : end + self.context_size]

        context = left + ' es ' + event + ' ee ' + right
        inputs.append(tokenizer.encode(context))
        print(context)

    inputs = pad_sequences(
      inputs,
      maxlen=self.max_length,
      dtype='long',
      truncating='post',
      padding='post')

    # create attention masks
    masks = []
    for seq in inputs:
      # use 1s for tokens and 0s for padding
      seq_mask = [float(i > 0) for i in seq]
      masks.append(seq_mask)

    return inputs, labels, masks

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  xml_dir = os.path.join(base, cfg.get('data', 'dev_xml'))
  text_dir = os.path.join(base, cfg.get('data', 'dev_text'))
  xml_regex = cfg.get('data', 'xml_regex')
  context_size = cfg.getint('args', 'context_size')

  dtr_data = DTRData(
    xml_dir,
    text_dir,
    xml_regex,
    cfg.getint('args', 'context_size'),
    cfg.getint('bert', 'max_len'))
  inputs, labels, masks = dtr_data()
