#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import random, argparse, os, shutil, torch
from transformers import (
    BertModel,
    BertPreTrainedModel,
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
import dataset_rel

# deterministic determinism
torch.manual_seed(2020)
random.seed(2020)

# new tokens to add to tokenizer
new_tokens = ['<t>', '</t>', '<e>', '</e>']

class BertWithSoftmax(BertPreTrainedModel):
  """Linear layer on top of pre-trained BERT"""

  def __init__(self, config):
    """Constructor"""

    super(BertWithSoftmax, self).__init__(config)

    self.bert = BertModel(config)
    self.softmax = torch.nn.LogSoftmax(1)

  def forward(self, input_ids, attention_mask):
    """Forward pass"""

    # (batch_size, seq_len, hidden_size=768)
    transformer_output = self.bert(input_ids, attention_mask)[0]

    # (batch_size, hidden_size=768)
    cls = transformer_output[:, 0, :]

    # all word pieces in vocabulary; (hidden_size=768, 30k)
    vocab = self.bert.embeddings.word_embeddings.weight.transpose(0, 1)

    # score [cls] against vocabulary; (batch_size, 30k)
    scores = torch.mm(cls, vocab)

    # apply softmax
    scores = self.softmax(scores)

    return scores

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
    lr=args.learning_rate,
    weight_decay=args.weight_decay)
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1500)

  return optimizer, scheduler

def fit(model, train_loader, val_loader):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = torch.nn.NLLLoss()
  optimizer, scheduler = make_optimizer_and_scheduler(model)

  optimal_epochs = 0
  best_loss = float('inf')

  for epoch in range(1, args.n_epochs + 1):
    train_loss, num_train_steps = 0, 0
    model.train()

    for batch in train_loader:
      optimizer.zero_grad()

      # metadata not needed here
      batch.pop('metadata')

      # tensors in batch to gpu
      for key in batch.keys():
        batch[key] = batch[key].to(device)

      # remove labels from batch
      batch_labels = batch.pop('labels')

      # model only needs inputs ids and attention masks
      batch_logits = model(**batch)
      loss = criterion(batch_logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss))

    # no early stopping, just save the model
    if not args.early_stopping:
      print('saving model after epoch %d...' % epoch)
      model.save_pretrained(args.model_dir)
      continue

    # early stopping, only save if loss decreased
    if val_loss < best_loss:
      print('loss improved, saving model...')
      model.save_pretrained(args.model_dir)
      best_loss = val_loss
      optimal_epochs = epoch

  return best_loss, optimal_epochs

def evaluate(model, data_loader):
  """Just compute the loss on the validation set"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  criterion = torch.nn.NLLLoss()
  total_loss, num_steps = 0, 0
  model.eval()

  for batch in data_loader:

    # metadata not needed here
    batch.pop('metadata')

    # tensors in batch to gpu
    for key in batch.keys():
      batch[key] = batch[key].to(device)

    # remove labels from batch
    batch_labels = batch.pop('labels')

    with torch.no_grad():
      batch_logits = model(**batch)
      loss = criterion(batch_logits, batch_labels)

    total_loss += loss.item()
    num_steps += 1

  average_loss = total_loss / num_steps
  return average_loss

def predict(model, data_loader, tokenizer):
  """Generate outputs for validation set samples"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  # relation arg id tuples
  pred_rels = []

  for batch in data_loader:

    # metadata for this batch
    metadata = batch.pop('metadata')

    for key in batch.keys():
      batch[key] = batch[key].to(device)

    # remove labels from batch
    batch_labels = batch.pop('labels')

    with torch.no_grad():
      batch_logits = model(**batch)

    inputs = tokenizer.batch_decode(
      batch['input_ids'],
      skip_special_tokens=True)
    batch_predictions = torch.argmax(batch_logits, dim=1)
    batch_labels = tokenizer.batch_decode(batch_labels)
    batch_predictions = tokenizer.batch_decode(batch_predictions)

    # iterate over samples in this batch
    for i in range(len(batch_predictions)):
      if args.print_predictions:
        print('[input]', inputs[i], '\n')
        print('[targets]', batch_labels[i], '\n')
        print('[predict]', batch_predictions[i], '\n')
      if args.print_errors:
        if batch['labels'][i].item() != batch_predictions[i].item():
          print('[input]', inputs[i], '\n')
          print('[targets]', batch_labels[i], '\n')
          print('[predict]', batch_predictions[i], '\n')
      if args.print_metadata:
        print('[metadata]', metadata[i], '\n')

      if len(metadata[i]) == 0:
        # no gold events or times in this chunk
        continue

      # parse metadata and map arg indexes to anafora ids
      arg_ind2anaf_id = {}
      for entry in metadata[i].split('||'):
        elements = entry.split('|')
        if len(elements) == 2:
          # no metadata lost due to length limitation
          arg_ind, anafora_id = elements
          arg_ind2anaf_id[arg_ind] = anafora_id

      # contained event or time (get it from the input)
      arg1 = inputs[i].split('|')[-1].lstrip()
      # container or none (get it from the output)
      arg2 = str(batch_predictions[i])

      # convert generated relations to anafora id pairs
      if arg1 in arg_ind2anaf_id and arg2 in arg_ind2anaf_id:
        pred_rels.append((arg_ind2anaf_id[arg1], arg_ind2anaf_id[arg2]))

  return pred_rels

def perform_fine_tuning():
  """Fine-tune and save model"""

  # need this to save a fine-tuned model
  if os.path.isdir(args.model_dir):
    shutil.rmtree(args.model_dir)
  os.mkdir(args.model_dir)

  # load pretrained bert tokenizer and add new tokens
  tokenizer = BertTokenizer.from_pretrained(args.model_name)
  tokenizer.add_tokens(new_tokens)

  model = BertWithSoftmax.from_pretrained(args.model_name)
  model.resize_token_embeddings(len(tokenizer))

  train_dataset = dataset_rel.Data(
    xml_dir=args.xml_train_dir,
    text_dir=args.text_train_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length)
  train_data_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.train_batch_size)

  test_dataset = dataset_rel.Data(
    xml_dir=args.xml_test_dir,
    text_dir=args.text_test_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length)
  val_data_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=args.train_batch_size)

  # fine-tune model on thyme data and save it
  best_loss, optimal_epochs = fit(
    model,
    train_data_loader,
    val_data_loader)
  print('best loss %.3f after %d epochs\n' % (best_loss, optimal_epochs))

def perform_evaluation():
  """Load fine-tuned model and generate"""

  # load pretrained bert tokenizer and add new tokens
  tokenizer = BertTokenizer.from_pretrained(args.model_name)
  tokenizer.add_tokens(new_tokens)

  # load a pretrained bert model
  model = BertWithSoftmax.from_pretrained(args.model_dir)
  model.resize_token_embeddings(len(tokenizer))

  test_dataset = dataset_rel.Data(
    xml_dir=args.xml_test_dir,
    text_dir=args.text_test_dir,
    out_dir=args.xml_out_dir,
    xml_regex=args.xml_regex,
    tokenizer=tokenizer,
    chunk_size=args.chunk_size,
    max_input_length=args.max_input_length)
  test_data_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=args.gener_batch_size)

  # make predictions using the saved model
  predicted_relations = predict(model, test_data_loader, tokenizer)
  print('writing xml...')
  test_dataset.write_xml(predicted_relations)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  arg_dict = dict(
    xml_train_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Train/'),
    text_train_dir=os.path.join(base, 'Thyme/Text/train/'),
    xml_test_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Dev/'),
    text_test_dir=os.path.join(base, 'Thyme/Text/dev/'),
    xml_out_dir='./Xml/',
    xml_regex='.*[.]Temporal.*[.]xml',
    model_dir='Model/',
    model_name='bert-base-uncased',
    chunk_size=50,
    max_input_length=512,
    n_files='all',
    learning_rate=5e-5,
    train_batch_size=48,
    gener_batch_size=64,
    weight_decay=0.01,
    print_predictions=False,
    print_metadata=False,
    print_errors=False,
    do_train=True,
    early_stopping=True,
    n_epochs=3)
  args = argparse.Namespace(**arg_dict)
  print('hyper-parameters: %s\n' % args)

  if args.do_train:
    print('starting training...')
    perform_fine_tuning()

  print('starting evaluation...')
  perform_evaluation()