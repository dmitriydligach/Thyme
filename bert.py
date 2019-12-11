#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch

from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import glob, os, logging, configparser

from dtrdata import DTRData

def performance_metrics(preds, labels):
  """Report performance metrics"""

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print(index, "->", f1)

def f1_micro(preds, labels):
  """Calculate the accuracy of our predictions vs labels"""

  predictions = np.argmax(preds, axis=1).flatten()
  f1 = f1_score(labels, predictions, average='micro')

  return f1

def make_data_loaders():
  """DataLoader(s) for train and dev sets"""

  xml_regex = cfg.get('data', 'xml_regex')

  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml'))
  train_text_dir = os.path.join(base, cfg.get('data', 'train_text'))

  dev_xml_dir = os.path.join(base, cfg.get('data', 'dev_xml'))
  dev_text_dir = os.path.join(base, cfg.get('data', 'dev_text'))

  train_data = DTRData(
    train_xml_dir,
    train_text_dir,
    xml_regex,
    cfg.getint('args', 'context_chars'),
    cfg.getint('bert', 'max_len'))
  dev_data = DTRData(
    dev_xml_dir,
    dev_text_dir,
    xml_regex,
    cfg.getint('args', 'context_chars'),
    cfg.getint('bert', 'max_len'))

  train_inputs, train_labels, train_masks = train_data()
  dev_inputs, dev_labels, dev_masks = dev_data()

  train_inputs = torch.tensor(train_inputs)
  dev_inputs = torch.tensor(dev_inputs)

  train_labels = torch.tensor(train_labels)
  dev_labels = torch.tensor(dev_labels)

  train_masks = torch.tensor(train_masks)
  dev_masks = torch.tensor(dev_masks)

  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)

  train_sampler = RandomSampler(train_data)
  dev_sampler = SequentialSampler(dev_data)

  train_data_loader = DataLoader(
    train_data,
    sampler=train_sampler,
    batch_size=cfg.getint('bert', 'batch_size'))
  dev_data_loader = DataLoader(
    dev_data,
    sampler=dev_sampler,
    batch_size=cfg.getint('bert', 'batch_size'))

  return train_data_loader, dev_data_loader

def main():
  """Fine-tune bert"""

  train_data_loader, dev_data_loader = make_data_loaders()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)

  model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4)
  if torch.cuda.is_available():
    model.cuda()
  else:
    model.cpu()

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() \
        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() \
        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
  optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=cfg.getfloat('bert', 'lr'),
    eps=1e-8)
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  # training loop
  for epoch in trange(cfg.getint('bert', 'num_epochs'), desc='epoch'):

    model.train()

    train_loss, num_train_examples, num_train_steps = 0, 0, 0

    for step, batch in enumerate(train_data_loader):

      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_masks, batch_labels = batch
      optimizer.zero_grad()

      loss, logits = model(
        batch_inputs,
        token_type_ids=None,
        attention_mask=batch_masks,
        labels=batch_labels)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_examples += batch_inputs.size(0)
      num_train_steps += 1

    print("epoch: {}, loss: {}".format(epoch, train_loss/num_train_steps))

    #
    # evaluation starts here ...
    #

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    num_eval_steps, num_eval_examples = 0, 0

    for batch in dev_data_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_masks, batch_labels = batch

      with torch.no_grad():
        [logits] = model(
          batch_inputs,
          token_type_ids=None,
          attention_mask=batch_masks)

      logits = logits.detach().cpu().numpy()
      label_ids = batch_labels.to('cpu').numpy()

      tmp_eval_accuracy = f1_micro(logits, label_ids)
      eval_accuracy += tmp_eval_accuracy
      num_eval_steps += 1

    print("validation accuracy: {}\n".format(eval_accuracy/num_eval_steps))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
