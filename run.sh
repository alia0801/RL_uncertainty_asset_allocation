#!/bin/bash
BASEDIR=$(dirname "$0")
echo "$BASEDIR"
for ag in 'ppo' 'td3'
do
    echo "Agent"
    echo "$ag"
    for code in {0..29}
    do
        echo "Portfolio code"
        echo "$code"
        /home/alia880801/anaconda3/envs/gpumorl/bin/python3.7 $BASEDIR/main.py  --code=$code --ag=$ag
    done
done