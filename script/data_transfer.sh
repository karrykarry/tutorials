#!/bin/bash

if [ $# != 1 ]; then
    echo "Please set the data you want to transfer"
    exit
fi

filename=$1
receving_side="donkey"

scp -r $filename $receving_side:~/yamcha_result/
