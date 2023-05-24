#!/usr/bin/env bash
DATAPATH="../data/raw_monolingual/indiccorp/data/hi/"
OUTPATH="../data/monolingual/training_splits/"

mkdir -p $OUTPATH
mkdir -p $OUTPATH"train/"
mkdir -p $OUTPATH"dev/"


test_size=15000

for f in $(ls $DATAPATH)
do
    cat $DATAPATH$f | head -n $test_size >  $OUTPATH"dev/hin.txt" ;
    cat $DATAPATH$f | head -n -$test_size > $OUTPATH"train/hin.txt" ;
done


