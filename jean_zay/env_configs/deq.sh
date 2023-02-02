#!/bin/bash
module purge
module load pytorch-gpu/py3/1.11.0

export TMP_DIR=$JOBSCRATCH/
export WIKITEXT_DIR=$SCRATCH/wikitext-103/
