#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from transformers import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

import numpy as np
import os, configparser, reldata

import metrics

class LstmClassifier(nn.Module):

  def __init__(self, hidden_size=1024, embed_dim=300, num_class=3):
    """Constructor"""

    super(LstmClassifier, self).__init__()
    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    self.embedding = nn.Embedding(tok.vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_size)
    self.dropout = nn.Dropout(0.25)
    self.linear = nn.Linear(hidden_size, num_class)

  def forward(self, texts):
    """Forward pass"""

    # embedding input: (batch_size, max_len)
    # embedding output: (batch_size, max_len, embed_dim)
    embeddings = self.embedding(texts)

    # lstm input: (seq_len, batch_size, input_size)
    # final state: (1, batch_size, hidden_size)
    embeddings = embeddings.permute(1, 0, 2)
    final_hidden, _ = self.lstm(embeddings)[1]

    # final hidden into (batch_size, hidden_size)
    final_hidden = final_hidden.squeeze()
    dropped = self.dropout(final_hidden)
    logits = self.linear(dropped)

    return logits

def to_inputs(texts, pad_token=0):
  """Converts texts into input matrices"""

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
  shape = (len(rows), max(len(row) for row in rows))
  token_ids = np.full(shape=shape, fill_value=pad_token)

  for i, row in enumerate(rows):
    token_ids[i, -len(row):] = row
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

  metrics.f1(
    all_labels,
    all_predictions,
    reldata.int2label,
    reldata.label2int)

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
