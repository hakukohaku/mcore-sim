#!/bin/bash

BATCH=(72)
DAVINCI_MICRO_BATCH=(32)
CIM_MICRO_BATCH=(1)

for batch in "${BATCH[@]}"; do
    for micro_batch in "${DAVINCI_MICRO_BATCH[@]}"; do
        echo "----------------Running batch $batch micro_batch $micro_batch arch DAVINCI----------------"
        make run BATCH=$batch MICRO_BATCH=$micro_batch ARCH=DAVINCI DP=2
        echo "------------------Finished----------------"
    done
    for micro_batch in "${CIM_MICRO_BATCH[@]}"; do
        echo "----------------Running batch $batch micro_batch $micro_batch arch CIM----------------"
        make run BATCH=$batch MICRO_BATCH=$micro_batch ARCH=CIM DP=4
        echo "------------------Finished----------------"
    done
done
python process_results.py --input_dir output/results --output output/all_results.csv