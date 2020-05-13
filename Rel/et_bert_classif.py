#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import numpy as np
import os, configparser, reldata

from sklearn.metrics import f1_score

class BertClassifier(BertPreTrainedModel):
  """Linear layer on top of pre-trained BERT"""

  def __init__(self, config):
    """Constructor"""

    super(BertClassifier, self).__init__(config)

    self.bert = BertModel(config)
    self.dropout = torch.nn.Dropout(0.1)
    self.linear = torch.nn.Linear(config.hidden_size, 3)

  def forward(self, input_ids, attention_mask):
    """Forward pass"""

    # (batch_size, seq_len, hidden_size=768)
    output = self.bert(input_ids, attention_mask)[0]
    # (batch_size, hidden_size=768)
    output = output[:, 0, :]
    output = self.dropout(output)
    logits = self.linear(output)

    return logits

def performance_metrics(labels, predictions):
  """Report performance metrics"""

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print('f1[%s] = %.3f' % (reldata.int2label[index], f1))

  ids = [reldata.label2int['CONTAINS'], reldata.label2int['CONTAINS-1']]
  contains_f1 = f1_score(labels, predictions, labels=ids, average='micro')
  print('f1[contains average] = %.3f' % contains_f1)

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

  token_ids = torch.tensor(token_ids)
  is_token = torch.tensor(is_token)

  return token_ids, is_token

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids, attention_masks = to_inputs(texts)
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, attention_masks, labels)
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

def train(bert_model, train_loader, device):
  """Training routine"""

  optimizer, scheduler = make_optimizer_and_scheduler(bert_model)

  for epoch in range(cfg.getint('bert', 'num_epochs')):
    bert_model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_masks, batch_labels = batch
      optimizer.zero_grad()

      logits = bert_model(batch_inputs, batch_masks)
      cross_entropy_loss = torch.nn.CrossEntropyLoss()
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    print('epoch: %d, loss: %.4f' % (epoch, train_loss / num_train_steps))

def evaluate(bert_model, data_loader, device):
  """Evaluation routine"""

  bert_model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_masks, batch_labels = batch

    with torch.no_grad():
      logits = bert_model(batch_inputs, batch_masks)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

  performance_metrics(all_labels, all_predictions)

  return all_predictions

def main():
  """Fine-tune bert"""

  bert_model = BertClassifier.from_pretrained(
    'bert-base-uncased',
    num_labels=3)

  if torch.cuda.is_available():
    device = torch.device('cuda')
    bert_model.cuda()
  else:
    device = torch.device('cpu')
    bert_model.cpu()

  train_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  texts, labels = train_data.event_time_relations()
  train_loader = make_data_loader(texts, labels, RandomSampler)

  train(bert_model, train_loader, device)

  dev_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  texts, labels = dev_data.event_time_relations()
  dev_loader = make_data_loader(texts, labels, sampler=SequentialSampler)

  evaluate(bert_model, dev_loader, device)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
