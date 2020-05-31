#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertPreTrainedModel

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import numpy as np
import os, configparser, random
import reldata, utils, metrics

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

class BertClassifier(BertPreTrainedModel):
  """Linear layer on top of pre-trained BERT"""

  def __init__(self, config, num_classes=2):
    """Constructor"""

    super(BertClassifier, self).__init__(config)

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(0.1)
    self.linear = nn.Linear(config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
    """Forward pass"""

    # (batch_size, seq_len, hidden_size=768)
    output = self.bert(input_ids, attention_mask)[0]
    # (batch_size, hidden_size=768)
    output = output[:, 0, :]
    output = self.dropout(output)
    logits = self.linear(output)

    return logits

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids, attention_masks = utils.to_bert_inputs(
    texts,
    max_len=cfg.getint('data', 'max_len'))
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('model', 'batch_size'))

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
    lr=cfg.getfloat('model', 'lr'))
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  return optimizer, scheduler

def train(model, train_loader, val_loader, weights):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = nn.CrossEntropyLoss(weights)

  optimizer, scheduler = make_optimizer_and_scheduler(model)

  for epoch in range(1, cfg.getint('model', 'num_epochs') + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_mask, batch_labels = batch
      optimizer.zero_grad()

      logits = model(batch_inputs, batch_mask)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    val_loss, f1 = evaluate(model, val_loader, weights)
    print('epoch: %d, train loss: %.3f, val loss: %.3f, val f1: %.3f' % \
          (epoch, train_loss / num_train_steps, val_loss, f1))

def evaluate(model, data_loader, weights, suppress_output=True):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = weights.to(device)
  model.to(device)

  cross_entropy_loss = nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_mask, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs, batch_mask)
      loss = cross_entropy_loss(logits, batch_labels)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

    total_loss += loss.item()
    num_steps += 1

  f1 = metrics.f1(all_labels,
                  all_predictions,
                  reldata.int2label,
                  reldata.label2int,
                  suppress_output)

  return total_loss / num_steps, f1

def main():
  """Fine-tune bert"""

  train_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  tr_texts, tr_labels = train_data.event_time_relations()
  train_loader = make_data_loader(tr_texts, tr_labels, RandomSampler)

  val_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  val_texts, val_labels = val_data.event_time_relations()
  val_loader = make_data_loader(val_texts, val_labels, SequentialSampler)

  model = BertClassifier.from_pretrained(
    'bert-base-uncased',
    num_labels=2)

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)
  evaluate(model, val_loader, weights, suppress_output=False)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
