# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the simulator runtime: architecture orchestration, core/DRAM/NoC behavior, shared types, and Pydantic config models. `tools/` generates instruction streams, while `run.py` is the main simulation entry point and `process_results.py` merges CSV outputs. Hardware, model, and power inputs live in `arch/`, `config/`, and `power/power_config/`. Use `tests/pipeline/` for workload JSON fixtures, `analysis/` for post-processing scripts, and treat `output/`, `log/`, and `data/` as generated artifacts.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`.

Run the default end-to-end flow with:
```bash
make run BATCH=1 MICRO_BATCH=1 DP=2 ARCH=CIM TECH=7
```
This generates a workload JSON, runs the simulator, and writes logs and CSV results.

Run the steps manually when debugging:
```bash
python tools/inst_generation.py -b 1 -mb 1 -dp 2 -o tests/pipeline/batch_1_micro_1_dp_2.json -a arch/cim.json -c config/cim.json
python run.py -b 1 -mb 1 -dp 2 -t 7 --arch_name CIM --arch arch/cim.json --fail failslow/normal.json --workload tests/pipeline/batch_1_micro_1_dp_2.json --power power/power_config/cim_power_7nm.json --output output/results/manual.csv
python process_results.py --input_dir output/results --output output/all_results.csv
```
Use `source results_gen.sh` for batch sweeps across architectures and micro-batch settings.
Use `make clean` only for disposable runs; it removes generated files from `output/`, `log/`, `tests/`, and `data/`.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, snake_case for functions/files, and PascalCase for Pydantic models and simulator classes. Match the existing import style and keep path handling rooted at the repository when adding new scripts. JSON config names are lowercase and architecture-specific, for example `arch/cim.json` and `power/power_config/davinci_power_22nm.json`. No formatter or linter is configured today, so keep changes small, readable, and consistent with surrounding code.

## Testing Guidelines
There is no formal `pytest` suite yet; validation is done with simulation smoke tests. Re-run `make run` or the exact `python run.py ...` command for the architecture you changed, then inspect the generated CSV in `output/results/`. When adding workload fixtures, follow the existing naming pattern `batch_<BATCH>_micro_<MICRO_BATCH>_dp_<DP>.json` under `tests/pipeline/`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `update readme`, `power config update`, and `pipeline with micro_batch and nonlinear`. Keep subjects brief and specific to one change. Pull requests should describe the scenario exercised, list modified config files, and include the command used for validation. Attach sample result CSVs or plots when behavior, performance, or power numbers change.
