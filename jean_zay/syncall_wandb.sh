#!/bin/bash

offline_runs = "$1/offline-run*"
while :
do
    for ofrun in $offline_runs
    do
        sleep 5m && wandb sync $ofrun;
    done
done