# Command-line interface for gaussian process emulator framework

import os
import subprocess
import sys
import argparse
import json
import pysvzerod as zerod

from gpe_dir import (gpe, gpe_config_handler, gpe_observables, gpe_param_handler, gpe_timeseries, run_zerod, misc_helpers) 

def cmd_inspect(args):
    with open(args.input_file, "r") as f:
        data = json.load(f)

    params = gpe_param_handler.discover_params(data)
    print(f"{len(params)} tunable parameters of given input file: \n")
    for param in params:
        print(f"\t Param: {param.id}, Baseline: {param.baseline}")

    with misc_helpers.quiet_func():
        results = zerod.simulate(data)

    vars = gpe_timeseries.list_variables(zerod.simulate(data))

    print(f"{len(vars)} output variables:")
    print(f"{var}" for var in vars) 




def cmd_run(args):
    if args.config:
        config = gpe_config_handler.ConfigHandler.load(args.config)
    else:
        config = gpe_config_handler.ConfigHandler(input_file=args.input_file)
    if args.default_bounds:
        config.default_bounds = args.default_bounds
    if args.param_bounds:
        for name, lower_bound, upper_bound in args.param_bounds:
            config.param_bounds[name] = (float(lower_bound), float(upper_bound))
    if args.num_samples:
        config.num_samples = args.num_samples
    if args.n_jobs:
        config.n_jobs = args.n_jobs
    if args.max_gpe_wave_reps:
        config.max_gpe_wave_reps = args.max_gpe_wave_reps
    if args.max_waves:
        config.max_waves = args.max_waves
    if args.samples_per_wave:
        config.samples_per_wave = args.samples_per_wave
    if args.gpe_cutoff:
        config.gpe_cutoff = args.gpe_cutoff

    output = gpe.history_match_wave(config) # output resulting from history match enabled gpe

    # implement plotting

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    cmd_parser = parser.add_subparsers(dest="cmd", required=True)

    inspect_parser = cmd_parser.add_parser(dest="inspect", help="Load input file and report (print) tunable parameters and output variables")
    inspect_parser.add_argument("input_file", help="Path to input model json file")
    inspect_parser.set_defaults(func=cmd_inspect)


    run_parser = cmd_parser.add_parser(dest="run", help="Run history matching enhanced gpe")
    run_parser.add_argument("input_file", help="Path to input model json file")
    run_parser.add_argument("--param_bounds", help="")
    run_parser.add_argument("--num_samples")
    run_parser.add_argument("--n_jobs")
    run_parser.add_argument("--max_gpe_wave_reps")
    run_parser.add_argument("--max_waves")
    run_parser.add_argument("--samples_per_wave")
    run_parser.add_argument("--gpe_cutoff")
    run_parser.add_argument("--calc_second_order")

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()