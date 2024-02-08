#!/bin/bash

#cd ~
#source execute.sh
 
python experiment.py -m 1 -K 4
python experiment.py -m 1 -K 5

python experiment.py -m 2 -K 4
python experiment.py -m 2 -K 5

python experiment.py -m 3 -K 4
python experiment.py -m 3 -K 5

python experiment.py -m 4 -K 4
python experiment.py -m 4 -K 5
