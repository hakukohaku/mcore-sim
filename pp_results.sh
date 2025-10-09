#!/bin/bash

BATCH=(60 72 84)
MICRO_BATCH=(8)

for batch in "${BATCH[@]}"; do
    for micro_batch in "${MICRO_BATCH[@]}"; do
        echo "----------------Running batch $batch micro_batch $micro_batch----------------" >> pp_results.txt
        make run BATCH=$batch MICRO_BATCH=$micro_batch > pp.log 2>&1
        tail -15 pp.log >> pp_results.txt
        echo "------------------Finished----------------" >> pp_results.txt
    done
done

echo "pp_results.txt is saved."