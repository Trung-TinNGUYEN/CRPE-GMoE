#!/bin/bash

#cd ~
#source execute_plots.sh

python params_to_metric_GLLiM.py -m 1 -K 4
python params_to_metric_GLLiM.py -m 1 -K 5

python plotting_GLLiM.py -m 1 -K 4
python plotting_GLLiM.py -m 1 -K 5

python params_to_metric_GLLiM.py -m 2 -K 4
python params_to_metric_GLLiM.py -m 2 -K 5

python plotting_GLLiM.py -m 2 -K 4
python plotting_GLLiM.py -m 2 -K 5

python params_to_metric_GLLiM.py -m 3 -K 4
python params_to_metric_GLLiM.py -m 3 -K 5

python plotting_GLLiM.py -m 3 -K 4
python plotting_GLLiM.py -m 3 -K 5

python params_to_metric_GLLiM.py -m 4 -K 4
python params_to_metric_GLLiM.py -m 4 -K 5

python plotting_GLLiM.py -m 4 -K 4
python plotting_GLLiM.py -m 4 -K 5

