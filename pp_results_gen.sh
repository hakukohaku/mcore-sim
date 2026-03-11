#!/bin/bash

BATCH=(72)
DAVINCI_MICRO_BATCH=(32)
CIM_MICRO_BATCH=(1)
TECH=(22)
for tech in "${TECH[@]}"; do
    for batch in "${BATCH[@]}"; do
        for micro_batch in "${DAVINCI_MICRO_BATCH[@]}"; do
            echo "----------------Running batch $batch micro_batch $micro_batch arch DAVINCI----------------"
            make run_pp BATCH=$batch MICRO_BATCH=$micro_batch ARCH=DAVINCI DP=2 TECH=$tech
            echo "------------------Finished----------------"
        done
        for micro_batch in "${CIM_MICRO_BATCH[@]}"; do
            echo "----------------Running batch $batch micro_batch $micro_batch arch CIM----------------"
            make run_pp BATCH=$batch MICRO_BATCH=$micro_batch ARCH=CIM DP=4 TECH=$tech
            echo "------------------Finished----------------"
        done
    done
done
python process_results.py --input_dir output/results --output output/all_results.csv