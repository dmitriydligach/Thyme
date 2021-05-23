#!/bin/bash

REF=Thyme/Official/thymedata/coloncancer/Dev/
PRED=../T5Rel/Xml/
CURRENT_DIR=$PWD

cd ../Anafora/

echo 'no closure'
python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*[.]Temporal.*[.]xml" -i TLINK:Type:CONTAINS

echo 'with closure'
python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*[.]Temporal.*[.]xml" -i TLINK:Type:CONTAINS --temporal-closure

cd $CURRENT_DIR
