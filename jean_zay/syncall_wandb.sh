#!/bin/bash

offline_runs="$1/offline-run*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
    sleep 5m
done