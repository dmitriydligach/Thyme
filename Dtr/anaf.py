#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import anafora, pytest

def test_add_entity():
    """Test testing testing"""

    data = anafora.AnaforaData()
    entity = anafora.AnaforaEntity()
    entity.id = '1@e@ID025_path_074@gold'
    data.annotations.append(entity)
    entity.type = 'EVENT'
    entity.parents_type = 'TemporalEntities'
    entity.properties['DocTimeRel'] = 'AFTER'

    data.indent()
    data.to_file('temp.xml')

if __name__ == "__main__":

  test_add_entity()
