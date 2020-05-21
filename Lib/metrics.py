#!/usr/bin/env python3

from sklearn.metrics import f1_score

def f1(labels, predictions, int2label, label2int, suppress_output=True):
  """Report performance metrics. Return the average."""

  f1 = f1_score(labels, predictions, average=None)
  for index, f1 in enumerate(f1):
    if not suppress_output:
      print('f1[%s] = %.3f' % (int2label[index], f1))

  ids = [label2int['CONTAINS'], label2int['CONTAINS-1']]
  contains_f1 = f1_score(labels, predictions, labels=ids, average='micro')
  if not suppress_output:
    print('f1[contains average] = %.3f' % contains_f1)

  return contains_f1

if __name__ == "__main__":

  print()
