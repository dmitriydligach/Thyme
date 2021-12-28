#!/bin/bash

REF=Thyme/Official/thymedata/coloncancer/Dev/
PRED=../GenRelCosine/Xml/
CURRENT_DIR=$PWD

cd ../Anafora/

# echo 'No closure:'
# python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*_clinic_.*[.]Temporal-Relation.*[.]xml" -i TLINK:Type:CONTAINS

# echo
# echo 'Closure:'

python -m anafora.evaluate -r $DATA_ROOT$REF -p $PRED -x ".*_clinic_.*[.]Temporal-Relation.*[.]xml" -i TLINK:Type:CONTAINS --temporal-closure

cd $CURRENT_DIR
