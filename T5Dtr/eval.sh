#!/bin/bash

REF=Thyme/Official/thymedata/coloncancer/Dev/
PRED=../T5Dtr/Xml/
CURRENT_DIR=$PWD

cd ../Anafora/

python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*[.]Temporal.*[.]xml" -i EVENT:DocTimeRel

cd $CURRENT_DIR
