#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
import transformers

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import glob, os, logging, configparser

import reldata

# required according to https://github.com/huggingface/transformers/issues/1350
class BERT(transformers.TFBertModel):
  def __init__(self, config, *inputs, **kwargs):
    super(BERT, self).__init__(config, *inputs, **kwargs)
    self.bert.call = tf.function(self.bert.call)

def to_inputs(texts, pad_token=0):
  """Converts texts into input matrices required by BERT"""

  tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

  rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
  shape = (len(rows), max(len(row) for row in rows))
  token_ids = np.full(shape=shape, fill_value=pad_token)
  is_token = np.zeros(shape=shape)

  for i, row in enumerate(rows):
    token_ids[i, :len(row)] = row
    is_token[i, :len(row)] = 1

  return dict(word_inputs=token_ids,
              mask_inputs=is_token,
              segment_inputs=np.zeros(shape=shape))

def get_model():
  """Model definition"""

  # Define inputs (token_ids, mask_ids, segment_ids)
  token_inputs = tf.keras.Input(shape=(None,), name='word_inputs', dtype='int32')
  mask_inputs = tf.keras.Input(shape=(None,), name='mask_inputs', dtype='int32')
  segment_inputs = tf.keras.Input(shape=(None,), name='segment_inputs', dtype='int32')

  # Load model and collect encodings
  bert = BERT.from_pretrained('bert-base-cased')
  token_encodings = bert([token_inputs, mask_inputs, segment_inputs])[0]

  # Keep only [CLS] token encoding
  sentence_encoding = tf.squeeze(token_encodings[:, 0:1, :], axis=1)

  # Apply dropout
  sentence_encoding = tf.keras.layers.Dropout(0.1)(sentence_encoding)

  # Final output layer
  outputs = tf.keras.layers.Dense(N_OUTPUTS, activation='sigmoid', name='outputs')(sentence_encoding)

  # Define model
  model = tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])

  return model

def performance_metrics(labels, predictions):
  """Report performance metrics"""

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print('f1[%s] = %.3f' % (reldata.int2label[index], f1))

  ids = [reldata.label2int['CONTAINS'], reldata.label2int['CONTAINS-1']]
  contains_f1 = f1_score(labels, predictions, labels=ids, average='micro')
  print('f1[contains average] = %.3f' % contains_f1)

def main():
  """Fine-tune bert"""

  train_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train')

  train_texts, train_labels = train_data.event_time_relations()
  train_texts = to_inputs(train_texts)

  model = get_model()
  model.fit(x=train_texts, y=train_labels, epochs=cfg.getint('data', 'num_epochs'))

  dev_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev')

  dev_texts, dev_labels = dev_data.event_time_relations()
  dev_texts = to_inputs(dev_texts)
  model.predict(dev_texts)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
