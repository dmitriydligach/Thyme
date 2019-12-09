#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import torch

from transformers import BertConfig, AdamW
from transformers import BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import numpy as np
import glob, os, logging, configparser

from dtrdata import DTRData

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# settings
gpu_num = 0
max_len = 512
batch_size = 8
epochs = 2

# scheduler
lr = 1e-3
max_grad_norm = 1.0
num_total_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)

def performance_metrics(preds, labels):
  """Report performance metrics"""

  predictions = np.argmax(preds, axis=1).flatten()
  f1 = f1_score(labels, predictions, average='macro')
  print('macro f1:', f1)

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    print(index, "->", f1)

def flat_accuracy(preds, labels):
  """Calculate the accuracy of our predictions vs labels"""

  # pred_flat = np.argmax(preds, axis=1).flatten()
  # labels_flat = labels.flatten()
  # return np.sum(pred_flat == labels_flat) / len(labels_flat)

  predictions = np.argmax(preds, axis=1).flatten()
  f1 = f1_score(labels, predictions, average='macro')

  return f1

def make_data_loaders():
  """DataLoader(s) for train and dev sets"""

  xml_regex = cfg.get('data', 'xml_regex')
  context_size = cfg.getint('args', 'context_size')

  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml'))
  train_text_dir = os.path.join(base, cfg.get('data', 'train_text'))

  dev_xml_dir = os.path.join(base, cfg.get('data', 'dev_xml'))
  dev_text_dir = os.path.join(base, cfg.get('data', 'dev_text'))

  train_data = DTRData(
    train_xml_dir,
    train_text_dir,
    xml_regex,
    context_size)
  dev_data = DTRData(
    dev_xml_dir,
    dev_text_dir,
    xml_regex,
    context_size)

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
    batch_size=batch_size)
  dev_data_loader = DataLoader(
    dev_data,
    sampler=dev_sampler,
    batch_size=batch_size)

  return train_data_loader, dev_data_loader

def main():
  """Fine-tune bert"""

  train_data_loader, dev_data_loader = make_data_loaders()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('device:', device)

  model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4)
  if torch.cuda.is_available():
    model.cuda()
  else:
    model.cpu()

  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]

  # this variable contains all of the hyperparemeter information our training loop needs
  optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
  scheduler = WarmupLinearSchedule(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_total_steps)

  # training loop
  for epoch in trange(epochs, desc="epoch"):
    model.train()

    # Tracking variables
    train_loss = 0
    num_train_examples = 0
    num_train_steps = 0

    # train for one epoch
    for step, batch in enumerate(train_data_loader):

      # add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_masks, batch_labels = batch
      optimizer.zero_grad()

      loss, logits = model(
        batch_inputs,
        token_type_ids=None,
        attention_mask=batch_masks,
        labels=batch_labels)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_examples += batch_inputs.size(0)
      num_train_steps += 1

    print("epoch: {}, loss: {}".format(epoch, train_loss/num_train_steps))

    #
    # evaluation starts here ...
    #

    # put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    num_eval_steps, num_eval_examples = 0, 0

    # evaluate data for one epoch
    for batch in dev_data_loader:

      # add batch to GPU
      batch = tuple(t.to(device) for t in batch)

      batch_inputs, batch_masks, batch_labels = batch

      with torch.no_grad():
        # forward pass; only logits returned since labels not provided
        [logits] = model(
          batch_inputs,
          token_type_ids=None,
          attention_mask=batch_masks)

      # move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = batch_labels.to('cpu').numpy()

      # print('logits:', logits)
      # print('label_ids:', label_ids)
      # print('logits shape:', logits.shape)
      # print('label_ids shape:', label_ids.shape)

      tmp_eval_accuracy = flat_accuracy(logits, label_ids)

      eval_accuracy += tmp_eval_accuracy
      num_eval_steps += 1

    print("validation accuracy: {}\n".format(eval_accuracy/num_eval_steps))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
