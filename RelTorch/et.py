#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from tqdm import trange
import numpy as np
import os, configparser, reldata

from sklearn.metrics import f1_score

def performance_metrics(labels, predictions):
  """Report performance metrics"""

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print('f1[%s] = %.3f' % (reldata.int2label[index], f1))

  ids = [reldata.label2int['CONTAINS'], reldata.label2int['CONTAINS-1']]
  contains_f1 = f1_score(labels, predictions, labels=ids, average='micro')
  print('f1[contains average] = %.3f' % contains_f1)

def evaluate(model, data_loader, device):
  """Model evaluation"""

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_masks, batch_labels = batch

    with torch.no_grad():
      [logits] = model(batch_inputs, attention_mask=batch_masks)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

  performance_metrics(all_labels, all_predictions)

  return all_predictions

def to_inputs(texts, pad_token=0):
  """Converts texts into input matrices required by BERT"""

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
  shape = (len(rows), max(len(row) for row in rows))
  token_ids = np.full(shape=shape, fill_value=pad_token)
  is_token = np.zeros(shape=shape)

  for i, row in enumerate(rows):
    token_ids[i, :len(row)] = row
    is_token[i, :len(row)] = 1

  return token_ids, is_token, np.zeros(shape=shape)

def make_data_loader(data_provider, sampler):
  """DataLoader objects for train or dev/test sets"""

  texts, labels = data_provider.event_time_relations()

  inputs, masks, _ = to_inputs(texts)

  inputs = torch.tensor(inputs)
  labels = torch.tensor(labels)
  masks = torch.tensor(masks)

  tensor_dataset = TensorDataset(inputs, masks, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('bert', 'batch_size'))

  return data_loader

def make_data_loader_old(data_provider, sampler):
  """DataLoader objects for train or dev/test sets"""

  texts, labels = data_provider.event_time_relations()

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  inputs = [tokenizer.encode(text) for text in texts]
  inputs = pad_sequences(
    inputs,
    maxlen=max([len(seq) for seq in inputs]),
    dtype='long',
    truncating='post',
    padding='post')
  masks = []  # attention masks
  for sequence in inputs:
    mask = [float(value > 0) for value in sequence]
    masks.append(mask)

  inputs = torch.tensor(inputs)
  labels = torch.tensor(labels)
  masks = torch.tensor(masks)

  tensor_dataset = TensorDataset(inputs, masks, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('bert', 'batch_size'))

  return data_loader

def make_optimizer_and_scheduler(model):
  """This is still a mystery to me"""

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() \
        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() \
        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
  optimizer = AdamW(
    params=optimizer_grouped_parameters,
    lr=cfg.getfloat('bert', 'lr'))
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  return optimizer, scheduler

def main():
  """Fine-tune bert"""

  model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3)

  if torch.cuda.is_available():
    device = torch.device('cuda')
    model.cuda()
  else:
    device = torch.device('cpu')
    model.cpu()

  optimizer, scheduler = make_optimizer_and_scheduler(model)

  train_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train')
  train_loader = make_data_loader(train_data, RandomSampler)

  for epoch in trange(cfg.getint('bert', 'num_epochs'), desc='epoch'):
    model.train()

    train_loss, num_train_examples, num_train_steps = 0, 0, 0

    for step, batch in enumerate(train_loader):
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_masks, batch_labels = batch
      optimizer.zero_grad()

      loss, logits = model(
        batch_inputs,
        attention_mask=batch_masks,
        labels=batch_labels)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_examples += batch_inputs.size(0)
      num_train_steps += 1

    print('epoch: %d, loss: %.4f' % (epoch, train_loss / num_train_steps))

  dev_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev')
  dev_loader = make_data_loader(dev_data, sampler=SequentialSampler)
  evaluate(model, dev_loader, device)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
