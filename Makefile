.PHONY: run test clean

B ?= 0
E ?= 200000
CURRENT_DIR := $(shell pwd)
OUTPUT_DIR := $(CURRENT_DIR)/output
BATCH ?= 1
MICRO_BATCH ?= 1
DP ?= 2
ARCH ?= CIM
INST_STREAM ?= $(CURRENT_DIR)/tests/pipeline/batch_$(BATCH)_micro_$(MICRO_BATCH)_channel_$(CHANNEL)_dp_$(DP).json

ifeq ($(wildcard $(OUTPUT_DIR)),)
create_folder:
	@echo "Creating $(OUTPUT_DIR) folder"
	@mkdir -p $(OUTPUT_DIR)
endif

ifeq ($(ARCH), CIM)
ARCHITECTURE = $(CURRENT_DIR)/arch/cim.json
POWER = $(CURRENT_DIR)/power/power_config/cim_power.json
else 
ARCHITECTURE = $(CURRENT_DIR)/arch/davinci.json
POWER = $(CURRENT_DIR)/power/power_config/davinci_power.json
endif

CONFIG ?= $(CURRENT_DIR)/config/davinci.json
FAIL ?= $(CURRENT_DIR)/failslow/normal.json
LOG ?= $(CURRENT_DIR)/log/pipeline_test.txt
OUTPUT ?= $(OUTPUT_DIR)/results/results_arch_$(ARCH)_batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP).csv

run:
	python $(CURRENT_DIR)/tools/test_pp.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug  --output $(OUTPUT) > $(OUTPUT_DIR)/run.log 2>&1

test:
	# python $(CURRENT_DIR)/test.py

clean:
	rm -rf gen
	rm -rf build
	rm -rf $(OUTPUT_DIR)
	rm -f run.log
