#!/usr/bin/env sh

PYTHON=$WORKON_HOME/m/bin/python

code=functions2d.py

outfile=${code%.*}_results.csv

$PYTHON $code > $outfile

mpc play
