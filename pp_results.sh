#!/bin/bash

BATCH=(60 72 84)
MICRO_BATCH=(32 64)

for batch in "${BATCH[@]}"; do
    for micro_batch in "${MICRO_BATCH[@]}"; do
        echo "----------------Running batch $batch micro_batch $micro_batch----------------" >> davinci_results.txt
        make run BATCH=$batch MICRO_BATCH=$micro_batch ARCH=DAVINCI
        echo "------------------Finished----------------" >> davinci_results.txt
    done
done
python process_results.py --input_dir output/results --output output/all_results.csv
echo "davinci_results.txt is saved."