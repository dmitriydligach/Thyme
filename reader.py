#!/usr/bin/env python3

import sys
sys.path.append('/Users/Dima/Loyola/Thyme/Git/anaforatools/')

import os
import anafora

train_xml = '/Users/Dima/Loyola/Data/Thyme/Official/thymedata/coloncancer/Train/'
train_text = '/Users/Dima/Loyola/Data/Thyme/Text/train/'
xml_regex = 'Temporal-Relation.gold.completed.xml'

def main():
  """Main Driver"""

  for sub_dir, text_name, file_names in anafora.walk(train_xml, xml_regex):
    xml_path = os.path.join(train_xml, sub_dir, file_names[0])
    ref_data = anafora.AnaforaData.from_file(xml_path)

    text_path = os.path.join(train_text, text_name)
    text = open(text_path).read()
    
    for data in ref_data.annotations.select_type('EVENT'):
      start, end = data.spans[0]
      context = text[start-20:end+20].replace('\n', '')
      event = text[start:end]
      dtr = data.properties['DocTimeRel']
      print('{}|{}|{}'.format(dtr, event, context))
      
if __name__ == "__main__":

  main()
  
