#!/usr/bin/env python3

import os, glob
from cassis import *

sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

def validate_xmi():
  """Extract events and times"""

  for xmi_path in glob.glob('/Users/Dima/Work/Data/Thyme/Xmi/*.xmi'):
    print(xmi_path)
    xmi_file = open(xmi_path, 'rb')
    type_system_file = open('/Users/Dima/Work/PyCharm/Thyme/T5Dtr/TypeSystem.xml', 'rb')
    type_system = load_typesystem(type_system_file)
    cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
    sys_view = cas.get_view('_InitialView')

    # iterate over sentences extracting events and times
    for sent in sys_view.select(sent_type):
      sent_text = sent.get_covered_text().replace('\n', '')
      print(sent_text)

if __name__ == "__main__":
  """My main man"""

  validate_xmi()