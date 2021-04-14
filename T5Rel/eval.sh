#!/bin/bash

REF=Thyme/Official/thymedata/coloncancer/Dev/
PRED=../T5Rel/Xml/
CURRENT_DIR=$PWD

cd ../Anafora/

python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*[.]Temporal.*[.]xml" -i TLINK:Type:CONTAINS

cd $CURRENT_DIR
