#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import BertTokenizer

import numpy as np
import os, configparser, math, random

import reldata, metrics, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

class TransformerClassifier(nn.Module):
  """A transformative experience"""

  def __init__(self, num_classes=3):
    """We have some of the best constructors in the world"""

    super(TransformerClassifier, self).__init__()

    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tok.vocab_size # the reason we need bert tokenizer

    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.position = PositionalEncoding(
      embedding_dim=cfg.getint('model', 'emb_dim'))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=cfg.getint('model', 'emb_dim'),
      nhead=cfg.getint('model', 'num_heads'),
      dim_feedforward=cfg.getint('model', 'feedforw_dim'))

    self.trans_encoder = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=cfg.getint('model', 'num_layers'))

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.linear = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=num_classes)

  def forward(self, texts):
    """Moving forward"""

    sqrtn = math.sqrt(cfg.getint('model', 'emb_dim'))
    output = self.embedding(texts) * sqrtn
    output = self.position(output)

    # encoder input: (seq_len, batch_size, emb_dim)
    # encoder output: (seq_len, batch_size, emb_dim)
    output = output.permute(1, 0, 2)
    output = self.trans_encoder(output)
    output = output[0, :, :]

    output = self.dropout(output)
    output = self.linear(output)

    return output

class PositionalEncoding(nn.Module):
  """That's my position"""

  def __init__(self, embedding_dim):
    """Deconstructing the construct"""

    super(PositionalEncoding, self).__init__()

    self.dropout = nn.Dropout(p=0.1)

    max_len = cfg.getint('data', 'max_len')
    pe = torch.zeros(max_len, embedding_dim)

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * \
                         (-math.log(10000.0) / embedding_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe)

  def forward(self, x):
    """We're being very forward here"""

    x = x + self.pe[:x.size(0), :]

    return self.dropout(x)

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids, attention_masks = utils.to_inputs(
    texts,
    cfg.getint('data', 'max_len'))
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    dataset=tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('model', 'batch_size'))

  return data_loader

def train(model, train_loader, dev_loader, device):
  """Training routine"""

  cross_entropy_loss = torch.nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  for epoch in range(cfg.getint('model', 'num_epochs')):
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
    f1 = evaluate(model, dev_loader, device)
    print('epoch: %d, train loss: %.3f, dev f1: %.3f' % \
          (epoch, av_loss, f1))

def evaluate(model, data_loader, device, suppress_output=True):
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

  f1 = metrics.f1(all_labels,
                  all_predictions,
                  reldata.int2label,
                  reldata.label2int,
                  suppress_output)

  return f1

def main():
  """Fine-tune bert"""

  model = TransformerClassifier()

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
  tr_texts, tr_labels = train_data.event_time_relations()
  train_loader = make_data_loader(tr_texts, tr_labels, RandomSampler)

  dev_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  dev_texts, dev_labels = dev_data.event_time_relations()
  dev_loader = make_data_loader(dev_texts, dev_labels, SequentialSampler)

  train(model, train_loader, dev_loader, device)
  evaluate(model, dev_loader, device, suppress_output=False)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
