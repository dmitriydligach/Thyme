#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from tokenizers import CharBPETokenizer
from sklearn.metrics import f1_score

import numpy as np
import os, configparser, random

import dtrdata, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

class BagOfEmbeddings(nn.Module):

  def __init__(self, num_class=4):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    tokenizer = CharBPETokenizer(
      '../Tokenize/thyme-tokenizer-vocab.json',
      '../Tokenize/thyme-tokenizer-merges.txt')
    vocab_size = tokenizer.get_vocab_size()

    self.embed = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.hidden1 = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=cfg.getint('model', 'hidden_size'))

    self.relu = nn.ReLU()

    self.hidden2 = nn.Linear(
      in_features=cfg.getint('model', 'hidden_size'),
      out_features=cfg.getint('model', 'hidden_size'))

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.classif = nn.Linear(
      in_features=cfg.getint('model', 'hidden_size'),
      out_features=num_class)

  def forward(self, texts):
    """Forward pass"""

    output = self.embed(texts)
    output = torch.mean(output, dim=1)
    output = self.hidden1(output)
    output = self.relu(output)
    output = self.hidden2(output)
    output = self.relu(output)
    output = self.dropout(output)
    output = self.classif(output)

    return output

def train(model, train_loader, val_loader, weights):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  for epoch in range(1, cfg.getint('model', 'num_epochs') + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_labels = batch
      optimizer.zero_grad()

      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, f1 = evaluate(model, val_loader, weights)
    print('epoch: %d, train loss: %.3f, val loss: %.3f, val f1: %.3f' % \
          (epoch, av_loss, val_loss, f1))

def evaluate(model, data_loader, weights):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = weights.to(device)
  model.to(device)

  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

    total_loss += loss.item()
    num_steps += 1

  f1 = f1_score(all_labels, all_predictions, average='micro')
  return total_loss / num_steps, f1

def main():
  """Fine-tune bert"""

  train_data = dtrdata.DTRData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  tr_texts, tr_labels = train_data.read()
  train_loader = utils.make_data_loader(
    tr_texts,
    tr_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'train',
    utils.to_token_id_sequences)

  val_data = dtrdata.DTRData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  val_texts, val_labels = val_data.read()
  val_loader = utils.make_data_loader(
    val_texts,
    val_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'dev',
    utils.to_token_id_sequences)

  model = BagOfEmbeddings()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)
  evaluate(model, val_loader, weights)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
