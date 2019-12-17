#!/bin/bash

REF=Thyme/Official/thymedata/coloncancer/Dev/
PRED=../Dtr/Xml/
CURRENT_DIR=$PWD

cd ../Anafora/

python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x "(?i).*clin.*Temp.*[.]xml$"

cd $CURRENT_DIR
