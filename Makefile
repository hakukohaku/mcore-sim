.PHONY: run test clean

CURRENT_DIR := $(shell pwd)
OUTPUT_DIR := $(CURRENT_DIR)/output
BATCH ?= 1
MICRO_BATCH ?= 1
DP ?= 2
ARCH ?= CIM
INST_STREAM ?= $(CURRENT_DIR)/tests/pipeline/batch_$(BATCH)_micro_$(MICRO_BATCH)_channel_$(CHANNEL)_dp_$(DP).json

CREATE_OUTPUT_DIR = @mkdir -p $(OUTPUT_DIR) $(OUTPUT_DIR)/results


ifeq ($(ARCH), CIM)
ARCHITECTURE = $(CURRENT_DIR)/arch/cim.json
POWER = $(CURRENT_DIR)/power/power_config/cim_power_7nm.json
CONFIG = $(CURRENT_DIR)/config/cim.json
else 
ARCHITECTURE = $(CURRENT_DIR)/arch/davinci.json
POWER = $(CURRENT_DIR)/power/power_config/davinci_power_7nm.json
CONFIG = $(CURRENT_DIR)/config/davinci.json
endif

FAIL ?= $(CURRENT_DIR)/failslow/normal.json
LOG ?= $(CURRENT_DIR)/log/$(ARCH)_batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP).txt
OUTPUT ?= $(OUTPUT_DIR)/results/results_arch_$(ARCH)_batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP).csv

run:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug  --output $(OUTPUT) > $(OUTPUT_DIR)/run.log 2>&1


clean:
	rm -rf $(OUTPUT_DIR)/run.log
	rm -rf $(CURRENT_DIR)/log/*
	rm -rf $(CURRENT_DIR)/tests/*
	rm -rf $(CURRENT_DIR)/data/*
