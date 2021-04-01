#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import os
import anafora

class RelData:
  """Make x and y from raw data"""

  def __init__(
    self,
    xml_dir,
    text_dir,
    xml_regex):
    """Constructor"""

    self.xml_dir = xml_dir
    self.text_dir = text_dir
    self.xml_regex = xml_regex

  def read(self):
    """Make x, y etc."""

    for sub_dir, text_name, file_names in anafora.walk(self.xml_dir, self.xml_regex):

      xml_path = os.path.join(self.xml_dir, sub_dir, file_names[0])
      ref_data = anafora.AnaforaData.from_file(xml_path)

      text_path = os.path.join(self.text_dir, text_name)
      text = open(text_path).read()

      for rel in ref_data.annotations.select_type('TLINK'):
        source = rel.properties['Source']
        target = rel.properties['Target']
        label = rel.properties['Type']

        start, end = source.spans[0]
        source_text = text[start:end]

        start, end = target.spans[0]
        target_text = text[start:end]

        print('%s(%s, %s)' % (label, source_text, target_text))


if __name__ == "__main__":

  base = os.environ['DATA_ROOT']

  rel_data = RelData(
    xml_dir=os.path.join(base, 'Thyme/Official/thymedata/coloncancer/Train/'),
    text_dir=os.path.join(base, 'Thyme/Text/train/'),
    xml_regex='.*[.]Temporal.*[.]xml')

  rel_data.read()
