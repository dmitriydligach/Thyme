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

import reldata

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

def make_data_loader(data_provider, sampler):
  """DataLoader objects for train or dev/test sets"""

  inputs, labels, masks = data_provider.event_event_relations()

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

  train_data = reldata.RelData(os.path.join(base, cfg.get('data', 'xmi_dir')))
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
    partition='dev',
    xml_ref_dir=os.path.join(base, cfg.get('data', 'ref_xml_dir')),
    xml_out_dir=cfg.get('data', 'out_xml_dir'))

  dev_loader = make_data_loader(dev_data, sampler=SequentialSampler)
  predictions = evaluate(model, dev_loader, device)
  # dev_data.write(predictions)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
