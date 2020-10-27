#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from tokenizers import CharBPETokenizer
from transformers import get_linear_schedule_with_warmup

import numpy as np
import os, configparser, math, random, copy

import reldata, metrics, utils

from transformer import EncoderLayer

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

    tokenizer = CharBPETokenizer(
      '../Tokenize/thyme-tokenizer-vocab.json',
      '../Tokenize/thyme-tokenizer-merges.txt')
    vocab_size = tokenizer.get_vocab_size()

    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.position = PositionalEncoding(
      embedding_dim=cfg.getint('model', 'emb_dim'))

    trans_encoders = []
    for n in range(cfg.getint('model', 'num_layers')):
      trans_encoders.append(EncoderLayer(
        d_model=cfg.getint('model', 'emb_dim'),
        d_inner=cfg.getint('model', 'feedforw_dim'),
        n_head=cfg.getint('model', 'num_heads'),
        d_k=cfg.getint('model', 'emb_dim'),
        d_v=cfg.getint('model', 'emb_dim')))
    self.trans_encoders = nn.ModuleList(trans_encoders)

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.linear = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=num_classes)

    self.init_weights()

  def init_weights(self):
    """Initialize the weights"""

    self.embedding.weight.data.uniform_(-0.1, 0.1)

  def forward(self, texts, attention_mask):
    """Moving forward"""

    sqrtn = math.sqrt(cfg.getint('model', 'emb_dim'))
    output = self.embedding(texts) * sqrtn
    output = self.position(output)

    # encoder input: (batch_size, seq_len, emb_dim)
    # encoder output: (batch_size, seq_len, emb_dim)

    for trans_encoder in self.trans_encoders:
      output, _ = trans_encoder(output)

    # output, _ = self.trans_encoder(output)

    # extract CLS token only
    # output = output[0, :, :]

    # average pooling
    output = torch.mean(output, dim=1)

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

def train(model, train_loader, val_loader, weights):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000)

  for epoch in range(1, cfg.getint('model', 'num_epochs') + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_mask, batch_labels = batch
      batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)
      optimizer.zero_grad()

      logits = model(batch_inputs, batch_mask)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, f1 = evaluate(model, val_loader, weights)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val f1: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, f1))

def evaluate(model, data_loader, weights, suppress_output=True):
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
    batch_inputs, batch_mask, batch_labels = batch
    batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)

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
  train_loader = utils.make_data_loader(
    tr_texts,
    tr_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'train',
    utils.to_transformer_inputs)

  val_data = reldata.RelData(
    os.path.join(base, cfg.get('data', 'xmi_dir')),
    partition='dev',
    n_files=cfg.get('data', 'n_files'))
  val_texts, val_labels = val_data.event_time_relations()
  val_loader = utils.make_data_loader(
    val_texts,
    val_labels,
    cfg.getint('model', 'batch_size'),
    cfg.getint('data', 'max_len'),
    'dev',
    utils.to_transformer_inputs)

  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)
  evaluate(model, val_loader, weights, suppress_output=False)

def init_transformer(m: torch.nn.Module):
  """Jiacheng Zhang's transformer initialization wisdom"""

  for name, params in m.named_parameters():
    print('initializing:', name)

    if len(params.shape) >= 2:
      torch.nn.init.xavier_uniform_(params)
    else:
      if 'bias' in name:
        torch.nn.init.zeros_(params)
      else:
        torch.nn.init.uniform_(params)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
