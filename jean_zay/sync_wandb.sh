#!/bin/bash

while :
do
    sleep 5m && wandb sync $1;
done