#!/usr/bin/env bash

INDIR="data/formatted/"
lang="mag"
INFILE=$INDIR"/"$lang"_hin.txt"

ALIGN_DIR="alignments"

ALIGN_FILE=$ALIGN_DIR"/"$lang"2hin.txt"
LEX_DIR="lexicons/target2hin/"
OUTFILE=$LEX_DIR"/"$lang"2hin.json"

# For Hindi as source
# ALIGN_FILE=$ALIGN_DIR"/hin2"$lang".txt"
# LEX_DIR="lexicons/hin2target/"
# OUTFILE=$LEX_DIR"/2hin."$lang"json"
# Also make changes in read_alignments.py --- as indicated


FA_DIR="../../..//fast_align/build/./fast_align"

# rm -r -f $ALIGN_DIR $LEX_DIR
mkdir -p $ALIGN_DIR $LEX_DIR

for file in $(ls $INDIR)
do
    $FA_DIR -i $INFILE -v -o -d > $ALIGN_FILE
    #For Hindi as source:
    # $FA_DIR -i $INFILE -r -v -o -d > $ALIGN_FILE
    python3 read_alignments.py $INFILE $ALIGN_FILE $OUTFILE
done
