#!/usr/bin/env python3

import os, sys, configparser
sys.path.append('./Anafora')

from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

import anafora

event_start = 'es'
event_end = 'ee'
max_len = 512

class DTRData:
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex,
    context_size):
    """Constructor"""

    ids = []
    labels = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    for sub_dir, text_name, file_names in anafora.walk(xml_dir, xml_regex):
      xml_path = os.path.join(xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      text_path = os.path.join(text_dir, text_name)
      text = open(text_path).read()

      for event in ref_data.annotations.select_type('EVENT'):
        labels.append(event.properties['DocTimeRel'])

        start, end = event.spans[0]
        event = text[start:end]
        left = text[start-context_size:start]
        right = text[end:end+context_size]
        context = '[CLS] ' + left + ' [ES] ' + event + ' [EE] ' + right + ' [SEP]'

        tokenized = tokenizer.tokenize(context)
        ids.append(tokenizer.convert_tokens_to_ids(tokenized))

    ids = pad_sequences(ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

    # create attention masks
    attention_masks = []
    for seq in ids:
      # use 1s for tokens and 0s for padding
      seq_mask = [float(i > 0) for i in seq]
      attention_masks.append(seq_mask)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  xml_dir = os.path.join(base, cfg.get('data', 'train_xml'))
  text_dir = os.path.join(base, cfg.get('data', 'train_text'))
  xml_regex = cfg.get('data', 'xml_regex')
  context_size = cfg.getint('args', 'context_size')

  dtr_data = DTRData(xml_dir, text_dir, xml_regex, context_size)
