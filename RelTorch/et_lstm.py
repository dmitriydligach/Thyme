#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import numpy as np
import os, configparser, reldata
from sklearn.metrics import f1_score

class LstmClassifier(nn.Module):

  def __init__(self, hidden_size=512, embed_dim=128, num_class=3):
    """Constructor"""

    super(LstmClassifier, self).__init__()
    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    self.embedding = nn.Embedding(tok.vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_size)
    self.dropout = nn.Dropout(0.1)
    self.linear = nn.Linear(hidden_size, num_class)

    # self.init_weights()

  def forward(self, texts):
    """Forward pass"""

    # input: (batch, max_len)
    # output: (batch, max_len, embed_dim)
    embeds = self.embedding(texts)

    batch = embeds.shape[0]
    max_len = embeds.shape[1]
    embed_dim = embeds.shape[2]

    # input: (seq_len, batch, input_size)
    output, (h_n, c_n) = self.lstm(embeds.view(max_len, batch, embed_dim))

    dropped = self.dropout(h_n.squeeze())
    logits = self.linear(dropped)

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
  """Converts texts into input matrices"""

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
  shape = (len(rows), max(len(row) for row in rows))
  token_ids = np.full(shape=shape, fill_value=pad_token)

  for i, row in enumerate(rows):
    token_ids[i, :len(row)] = row
  token_ids = torch.tensor(token_ids)

  return token_ids

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids = to_inputs(texts)
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('bert', 'batch_size'))

  return data_loader

def train(model, train_loader, device):
  """Training routine"""

  cross_entropy_loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.getfloat('bert', 'lr'))
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

  for epoch in range(cfg.getint('bert', 'num_epochs')):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_labels = batch
      optimizer.zero_grad()

      logits = model(batch_inputs)

      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      optimizer.step()
      # scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    print('epoch: %d, loss: %.4f' % (epoch, train_loss / num_train_steps))

def evaluate(model, data_loader, device):
  """Evaluation routine"""

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

  performance_metrics(all_labels, all_predictions)

  return all_predictions

def main():
  """Fine-tune bert"""

  model = LstmClassifier()

  if torch.cuda.is_available():
    device = torch.device('cuda')
    model.cuda()
  else:
    device = torch.device('cpu')
    model.cpu()

  train_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  texts, labels = train_data.event_time_relations()
  train_loader = make_data_loader(texts, labels, RandomSampler)

  train(model, train_loader, device)

  dev_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  texts, labels = dev_data.event_time_relations()
  dev_loader = make_data_loader(texts, labels, SequentialSampler)

  evaluate(model, dev_loader, device)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
