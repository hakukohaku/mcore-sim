.PHONY: run test clean

B ?= 0
E ?= 2000000000000000000000000000000000000000000000000000000000000
BATCH ?= 1
MICRO_BATCH ?= 1
CHANNEL ?= 16
DP ?= 4
OUTPUT ?= tests/pipeline/batch_$(BATCH)_micro_$(MICRO_BATCH)_channel_$(CHANNEL)_dp_$(DP).json
# 下面这些变量可以在命令行传入，例如：make run FLOW=--flow
FLOW ?=


run:
	# python run.py --simstart=$(B) --simend=$(E) $(FLOW)
	python tools/test_pp.py -b $(BATCH) -m $(MICRO_BATCH) -c $(CHANNEL) -dp $(DP) -o $(OUTPUT)
	python run.py --arch arch/myarch_gemini4_4_cim.json --fail failslow/normal.json --workload $(OUTPUT) --log log/pipeline_test.txt --level debug > pp.log 2>&1 && tail -20 pp.log > results.txt

test:
	python test.py

results:
	@echo "提取模拟结果到 results.txt..."
	@tail -20 pp.log > results.txt
	@echo "结果已保存到 results.txt"

clean:
	rm -rf gen
	rm -rf build
	rm -f results.txt

