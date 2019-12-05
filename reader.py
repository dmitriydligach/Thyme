#!/usr/bin/env python3

import sys
sys.path.append('./Anafora/')

import os, configparser
import anafora

def main(xml_dir, text_dir, xml_regex):
  """Main Driver"""

  for sub_dir, text_name, file_names in anafora.walk(xml_dir, xml_regex):

    xml_path = os.path.join(xml_dir, sub_dir, file_names[0])
    ref_data = anafora.AnaforaData.from_file(xml_path)

    text_path = os.path.join(text_dir, text_name)
    text = open(text_path).read()

    for data in ref_data.annotations.select_type('EVENT'):
      start, end = data.spans[0]
      context = text[start-20:end+20].replace('\n', '')
      event = text[start:end]
      dtr = data.properties['DocTimeRel']
      print('{}|{}|{}'.format(dtr, event, context))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  xml_dir = os.path.join(base, cfg.get('data', 'train_xml'))
  text_dir = os.path.join(base, cfg.get('data', 'train_text'))
  xml_regex = cfg.get('data', 'xml_regex')

  main(xml_dir, text_dir, xml_regex)
