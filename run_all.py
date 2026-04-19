#!/usr/bin/env python3
"""
Run encoder and decoder simulation cases in parallel across multiple
architecture + optimization configurations.

Phase 1: Generate instruction JSON per (ARCH, case_name, input_len, ekv) — shared
          across configs that differ only in scaling flags.
Phase 2: Run simulations in parallel (all 8 processes).

Each --config entry is a comma-separated string: ARCH,flag1,flag2,...
Valid flags: complex, complex_opt, co_opt, single_tp

Usage:
    python run_all.py --tech 7 --batch 24 --dp 2 \
        --config "DAVINCI,complex" \
        --config "CIM,complex" \
        --config "CIM,complex,complex_opt" \
        --config "CIM,complex,complex_opt,co_opt"
"""

import argparse
import json
import os
import subprocess
import sys


VALID_FLAGS = {"complex", "complex_opt", "co_opt", "single_tp"}


def get_config_defaults(arch: str):
    config_path = f"config/{arch.lower()}.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    model = config["model"]
    tp_degree = 1
    if "pptp_config" in config:
        tp_degree = config["pptp_config"].get("tp", 1)
    elif "tp_config" in config:
        tp_degree = config["tp_config"].get("tp", 1)
    return model["input_len"], model.get("n_encoder_kv_len", 0), tp_degree


def parse_config(config_str: str):
    """Parse 'ARCH,flag1,flag2,...' into (arch, flags_set)."""
    parts = [p.strip() for p in config_str.split(",")]
    arch = parts[0].upper()
    if arch not in ("CIM", "DAVINCI"):
        raise ValueError(f"Unknown architecture: {arch}. Must be CIM or DAVINCI.")
    flags = set(parts[1:])
    unknown = flags - VALID_FLAGS
    if unknown:
        raise ValueError(f"Unknown flags: {unknown}. Valid: {VALID_FLAGS}")
    return arch, flags


def config_tag(arch: str, flags: set) -> str:
    """Generate a short tag for file naming, e.g. 'CIM_co_opt_complex_complex_opt'."""
    if not flags:
        return arch
    return arch + "_" + "_".join(sorted(flags))


def build_gen_cmd(arch, batch, micro_batch, dp, input_len, ekv, inst_stream,
                  single_tp=False):
    """Build make gen_pptp command."""
    cmd = [
        "make", "gen_pptp",
        f"BATCH={batch}", f"MICRO_BATCH={micro_batch}", f"DP={dp}",
        f"ARCH={arch}", f"N_ENCODER_KV_LEN={ekv}", f"INPUT_LEN={input_len}",
        f"INST_STREAM={inst_stream}",
    ]
    if single_tp:
        cmd.append("SINGLE_TP=1")
    return cmd


def build_sim_cmd(arch, tech, batch, micro_batch, dp, inst_stream,
                  output_path, tag, single_tp=False, tp_scale=1,
                  complex_enable=False, complex_opt_enable=False,
                  co_opt_enable=False, quiet=True):
    """Build make sim_pptp command (simulation only, no inst generation)."""
    log_path = f"log/{tag}_batch_{batch}_micro_{micro_batch}_dp_{dp}_tech_{tech}nm.txt"
    cmd = [
        "make", "sim_pptp",
        f"BATCH={batch}", f"MICRO_BATCH={micro_batch}", f"DP={dp}",
        f"ARCH={arch}", f"TECH={tech}",
        f"OUTPUT={output_path}",
        f"INST_STREAM={inst_stream}",
        f"LOG={log_path}",
    ]
    if quiet:
        cmd.append("QUIET=1")
    if single_tp:
        cmd.append("SINGLE_TP=1")
        cmd.append(f"TP_SCALE={tp_scale}")
    if complex_enable:
        cmd.append("COMPLEX_ENABLE=1")
    if complex_opt_enable:
        cmd.append("COMPLEX_OPT_ENABLE=1")
    if co_opt_enable:
        cmd.append("CO_OPT_ENABLE=1")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run encoder/decoder simulation cases across multiple arch+flag configs in parallel")

    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--micro_batch", type=int, default=1)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--tech", type=int, default=7, choices=[7, 22])

    parser.add_argument("--encoder_input_len", type=int, default=68)
    parser.add_argument("--encoder_ekv", type=int, default=0)
    parser.add_argument("--decoder_input_len", type=int, default=64)
    parser.add_argument("--decoder_ekv", type=int, default=68)

    parser.add_argument("--config", type=str, action="append", required=True,
                        help='Architecture + flags, e.g. "CIM,complex,complex_opt"')
    parser.add_argument("--output_dir", type=str, default="output/results")
    parser.add_argument("--verbose", action="store_true",
                        help="Keep log/power_trace/data output (default: quiet, CSV only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse all configurations
    configs = []
    for cfg_str in args.config:
        arch, flags = parse_config(cfg_str)
        configs.append((arch, flags))

    # Collect unique (arch, case, input_len, ekv) for instruction generation
    # and all simulation tasks
    gen_keys = {}  # (arch, case_name, il, ekv) -> inst_stream path
    sim_tasks = []

    for arch, flags in configs:
        _, _, tp_degree = get_config_defaults(arch)
        tag = config_tag(arch, flags)
        use_single_tp = "single_tp" in flags

        for case_name, il, ekv in [
            ("encoder", args.encoder_input_len, args.encoder_ekv),
            ("decoder", args.decoder_input_len, args.decoder_ekv),
        ]:
            # Instruction JSON depends only on arch + case params, not scaling flags
            gen_key = (arch, case_name, il, ekv, use_single_tp)
            if gen_key not in gen_keys:
                inst_stream = (f"tests/pipeline/{case_name}_{arch}"
                               f"_batch_{args.batch}_micro_{args.micro_batch}_dp_{args.dp}.json")
                gen_keys[gen_key] = inst_stream

            sim_tasks.append({
                "arch": arch, "flags": flags, "tag": tag,
                "case_name": case_name, "il": il, "ekv": ekv,
                "tp_degree": tp_degree, "single_tp": use_single_tp,
                "inst_stream": gen_keys[gen_key],
                "output": f"{args.output_dir}/{case_name}_{tag}.csv",
            })

    # ── Phase 1: Generate instruction JSONs (sequential, shared across configs) ──
    print(f"=== Phase 1: Generating {len(gen_keys)} instruction files ===")
    for (arch, case_name, il, ekv, single_tp), inst_stream in gen_keys.items():
        print(f"  [GEN] {case_name}({arch}): input_len={il}, ekv={ekv}")
        cmd = build_gen_cmd(arch, args.batch, args.micro_batch, args.dp,
                            il, ekv, inst_stream, single_tp=single_tp)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"  [FAIL] {case_name}({arch})\n{result.stderr.decode()}")
            sys.exit(1)
        print(f"  [OK]  -> {inst_stream}")

    # ── Phase 2: Run simulations in parallel ──
    print(f"\n=== Phase 2: Launching {len(sim_tasks)} simulations ===")
    processes = {}
    for t in sim_tasks:
        label = f"{t['case_name']}({t['tag']})"
        cmd = build_sim_cmd(
            t["arch"], args.tech, args.batch, args.micro_batch, args.dp,
            t["inst_stream"], t["output"], f"{t['case_name']}_{t['tag']}",
            single_tp=t["single_tp"], tp_scale=t["tp_degree"],
            complex_enable="complex" in t["flags"],
            complex_opt_enable="complex_opt" in t["flags"],
            co_opt_enable="co_opt" in t["flags"],
            quiet=not args.verbose,
        )
        print(f"  [START] {label}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes[label] = (proc, t["output"])

    print(f"\n{len(processes)} simulation processes running...\n")

    failed = False
    for label, (proc, output) in processes.items():
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            print(f"[FAIL] {label}\n{stderr}")
            failed = True
        else:
            print(f"[DONE] {label} -> {output}")

    if failed:
        sys.exit(1)
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()
