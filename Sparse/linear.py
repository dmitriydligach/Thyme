#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch

from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import glob, os, logging, configparser

import dtrdata

def tokenizer(string):
  """Custom tokenizer"""

  return string.split(' ')

def grid_search(x, y, scoring):
  """Find best model and fit it"""

  param_grid = {
    'penalty': ['l1', 'l2'],
    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
  lr = LogisticRegression(class_weight='balanced')
  gs = GridSearchCV(lr, param_grid, scoring=scoring, cv=10)
  gs.fit(x, y)

  return gs.best_estimator_

def get_data():
  """Load official THYME data"""

  xml_regex = cfg.get('data', 'xml_regex')

  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml'))
  train_text_dir = os.path.join(base, cfg.get('data', 'train_text'))

  dev_xml_dir = os.path.join(base, cfg.get('data', 'dev_xml'))
  dev_text_dir = os.path.join(base, cfg.get('data', 'dev_text'))

  train_data = dtrdata.DTRData(
    train_xml_dir,
    train_text_dir,
    xml_regex,
    cfg.getint('args', 'context_chars'))
  dev_data = dtrdata.DTRData(
    dev_xml_dir,
    dev_text_dir,
    xml_regex,
    cfg.getint('args', 'context_chars'))

  x_train, y_train = train_data()
  x_dev, y_dev = dev_data()

  vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None)
  x_train = vectorizer.fit_transform(x_train)
  x_dev = vectorizer.transform(x_dev)

  return x_train, y_train, x_dev, y_dev

def performance_metrics(labels, predictions):
  """Report performance metrics"""

  f1_micro = f1_score(labels, predictions, average='micro')
  print('f1[micro] = %.3f' % f1_micro)

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print('f1[%s] = %.3f' % (dtrdata.int2label[index], f1))

def eval(search=False):
  """Fine-tune bert"""

  x_train, y_train, x_dev, y_dev = get_data()

  if search:
    classifier = grid_search(x_train, y_train, 'f1_micro')
  else:
    classifier = LogisticRegression(class_weight='balanced')
    model = classifier.fit(x_train, y_train)

  predictions = classifier.predict(x_dev)
  performance_metrics(y_dev, predictions)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  eval()
