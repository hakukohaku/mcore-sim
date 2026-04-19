.PHONY: run test clean

CURRENT_DIR := $(shell pwd)
OUTPUT_DIR := $(CURRENT_DIR)/output
BATCH ?= 2
MICRO_BATCH ?= 1
DP ?= 2
PP ?= 8
N_ENCODER_KV_LEN ?=0
INPUT_LEN ?=
SINGLE_TP ?=
TP_SCALE ?=
COMPLEX_ENABLE ?=
COMPLEX_OPT_ENABLE ?=
CO_OPT_ENABLE ?=
ARCH ?= CIM
TECH ?= 7
INST_STREAM ?= $(CURRENT_DIR)/tests/pipeline/batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP)_$(ARCH).json

CREATE_OUTPUT_DIR = @mkdir -p $(OUTPUT_DIR) $(OUTPUT_DIR)/results


ifeq ($(ARCH), CIM)
ARCHITECTURE = $(CURRENT_DIR)/arch/cim.json
POWER = $(CURRENT_DIR)/power/power_config/cim_power_$(TECH)nm.json
CONFIG = $(CURRENT_DIR)/config/cim.json
else 
ARCHITECTURE = $(CURRENT_DIR)/arch/davinci.json
POWER = $(CURRENT_DIR)/power/power_config/davinci_power_$(TECH)nm.json
CONFIG = $(CURRENT_DIR)/config/davinci.json
endif

INPUT_LEN_FLAG = $(if $(INPUT_LEN),-il $(INPUT_LEN),)
SINGLE_TP_FLAG = $(if $(SINGLE_TP),-stp,)
TP_SCALE_FLAG = $(if $(TP_SCALE),--tp_scale $(TP_SCALE),)
COMPLEX_ENABLE_FLAG = $(if $(COMPLEX_ENABLE),--complex_enable,)
COMPLEX_OPT_ENABLE_FLAG = $(if $(COMPLEX_OPT_ENABLE),--complex_opt_enable,)
CO_OPT_ENABLE_FLAG = $(if $(CO_OPT_ENABLE),--co_opt_enable,)
QUIET ?=
QUIET_FLAG = $(if $(QUIET),--quiet,)

FAIL ?= $(CURRENT_DIR)/failslow/normal.json
LOG ?= $(CURRENT_DIR)/log/$(ARCH)_batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP)_tech_$(TECH)nm.txt
OUTPUT ?= $(OUTPUT_DIR)/results/results_arch_$(ARCH)_batch_$(BATCH)_micro_$(MICRO_BATCH)_dp_$(DP)_tech_$(TECH)nm.csv
RUN_LOG ?= $(OUTPUT_DIR)/run.log
POWER_TRACE ?= $(CURRENT_DIR)/power/power_trace/power_trace.txt
STDOUT_REDIRECT = $(if $(QUIET),/dev/null,$(RUN_LOG))

run_pp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/pp_inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -t $(TECH) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug --output $(OUTPUT) --power_trace $(POWER_TRACE) $(QUIET_FLAG) $(COMPLEX_ENABLE_FLAG) $(COMPLEX_OPT_ENABLE_FLAG) $(CO_OPT_ENABLE_FLAG) > $(STDOUT_REDIRECT) 2>&1

run_tp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/tp_inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -pp $(PP) -ekv $(N_ENCODER_KV_LEN) $(INPUT_LEN_FLAG) $(SINGLE_TP_FLAG) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -t $(TECH) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug $(TP_SCALE_FLAG) --output $(OUTPUT) --power_trace $(POWER_TRACE) $(QUIET_FLAG) $(COMPLEX_ENABLE_FLAG) $(COMPLEX_OPT_ENABLE_FLAG) $(CO_OPT_ENABLE_FLAG) > $(STDOUT_REDIRECT) 2>&1

run_pptp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/pptp_inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -ekv $(N_ENCODER_KV_LEN) $(INPUT_LEN_FLAG) $(SINGLE_TP_FLAG) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -t $(TECH) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug $(TP_SCALE_FLAG) --output $(OUTPUT) --power_trace $(POWER_TRACE) $(QUIET_FLAG) $(COMPLEX_ENABLE_FLAG) $(COMPLEX_OPT_ENABLE_FLAG) $(CO_OPT_ENABLE_FLAG) > $(STDOUT_REDIRECT) 2>&1

gen_pptp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/pptp_inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -ekv $(N_ENCODER_KV_LEN) $(INPUT_LEN_FLAG) $(SINGLE_TP_FLAG) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)

sim_pptp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/run.py -b $(BATCH) -mb $(MICRO_BATCH) -dp $(DP) -t $(TECH) --arch_name $(ARCH) --arch $(ARCHITECTURE) --fail $(FAIL) --workload $(INST_STREAM) --power $(POWER) --log $(LOG) --level debug $(TP_SCALE_FLAG) --output $(OUTPUT) --power_trace $(POWER_TRACE) $(QUIET_FLAG) $(COMPLEX_ENABLE_FLAG) $(COMPLEX_OPT_ENABLE_FLAG) $(CO_OPT_ENABLE_FLAG) > $(STDOUT_REDIRECT) 2>&1

gen_tp:
	$(CREATE_OUTPUT_DIR)
	python $(CURRENT_DIR)/tools/tp_inst_generation.py -b $(BATCH) -mb $(MICRO_BATCH) -pp $(PP) -ekv $(N_ENCODER_KV_LEN) $(INPUT_LEN_FLAG) -dp $(DP) -o $(INST_STREAM) -a $(ARCHITECTURE) -c $(CONFIG)

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf $(CURRENT_DIR)/log/*
	rm -rf $(CURRENT_DIR)/tests/*
	rm -rf $(CURRENT_DIR)/data/*
	rm -rf $(CURRENT_DIR)/power/power_trace/*
