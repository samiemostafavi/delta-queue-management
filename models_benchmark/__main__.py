import getopt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

from .gym import parse_gym_args, run_gym_processes
from .newdelta import add_delta_params, run_newdelta
from .noaqm import run_noaqm
from .offlineoptimum import add_offlineoptimum_params, run_offlineoptimum
from .plot import plot_main
from .train import parse_train_args, run_train_processes


def process(params, return_dict, main_benchmark: Callable, main_benchmark_name: str):

    if params["run_noaqm"]:
        logger.info(f"{params['run_number']}: Running noaqm")
        run_noaqm(params, return_dict, "noaqm")
        params["delay_bound"] = return_dict[params["run_number"]]["quantile_value"]
        logger.info(f"Set delay_bound={params['delay_bound']}")
    else:
        logger.info(f"{params['run_number']}: Not running noaqm, reading results")
        noaqm_res_path = (
            params["records_path"]
            + "noaqm/"
            + str(params["run_number"])
            + "_noaqm_result.json"
        )

        # Opening result json file
        with open(noaqm_res_path) as json_file:
            noaqm_res = json.load(json_file)

        logger.info(
            f"Loaded noaqm {params['run_number']}: "
            + f"quantile_value={noaqm_res['quantile_value']}, "
        )

        params["delay_bound"] = noaqm_res["quantile_value"]
        logger.info(f"Set delay_bound={params['delay_bound']}")

    logger.info(f"{params['run_number']}: Running {main_benchmark_name}")
    main_benchmark(params, return_dict, main_benchmark_name)


def parse_run_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    args_dict["run-noaqm"] = False
    try:
        opts, args = getopt.getopt(
            argv,
            "ha:u:l:m:n",
            [
                "arrival-rate=",
                "until=",
                "label=",
                "model=",
                "run-noaqm",
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m predictors_benchmark run -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m predictors_benchmark run "
                + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-a", "--arrival-rate"):
            args_dict["arrival-rate"] = float(arg)
        elif opt in ("-u", "--until"):
            args_dict["until"] = int(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-m", "--model"):
            if arg == "gmm":
                args_dict["predictor_type"] = "gmm"
                args_dict["module_set_param"] = add_delta_params
                args_dict["module_run"] = run_newdelta
                args_dict["module_label"] = "gmm"
            elif arg == "gmevm":
                args_dict["predictor_type"] = "gmevm"
                args_dict["module_set_param"] = add_delta_params
                args_dict["module_run"] = run_newdelta
                args_dict["module_label"] = "gmevm"
            elif arg == "offline-optimum":
                args_dict["module_set_param"] = add_offlineoptimum_params
                args_dict["module_run"] = run_offlineoptimum
                args_dict["module_label"] = "offline-optimum"
            else:
                raise Exception("wrong module name")
        elif opt in ("-n", "--run-noaqm"):
            args_dict["run-noaqm"] = True

    return args_dict


def parse_plot_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hp:m:t:",
            ["project=", "models=", "type="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m predictors_benchmark -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m delay_bound_benchmark plot "
                + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-p", "--project"):
            # project folder setting
            p = Path(__file__).parents[0]
            args_dict["project_folder"] = str(p) + "/" + arg + "_results/"
        elif opt in ("-m", "--models"):
            args_dict["models"] = [s.strip() for s in arg.split(",")]
        elif opt in ("-t", "--type"):
            args_dict["type"] = arg

    return args_dict


def run_processes(exp_args: dict):
    # this function creates the param dict for the run

    logger.info(
        "Prepare delay-bound benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # experiment parameters
    # quantile values of no-aqm model with p1 as gpd_concentration
    bench_params = {  # target_delay
        "p999": 0.999,
        "p99": 0.99,
        "p9": 0.9,
        "p8": 0.8,
    }

    manager = mp.Manager()
    return_dict = manager.dict()

    # 4 x 4, until 1000000 took 7 hours
    sequential_runs = 1  # 2  # 2  # 4
    parallel_runs = 16  # 8  # 8  # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())
            key_this_run = keys[i % len(keys)]

            # create and prepare the results directory
            records_path = project_path + key_this_run + "/"
            os.makedirs(records_path, exist_ok=True)

            # save the json info
            jsoninfo = json.dumps({"quantile_key": bench_params[key_this_run]})
            with open(records_path + "info.json", "w") as f:
                f.write(jsoninfo)

            run_number = j * parallel_runs + i
            params = {
                "main_path": main_path,
                "project_path": project_path,
                "records_path": records_path,
                "run_noaqm": exp_args["run-noaqm"],
                "arrivals_number": 1000000,  # 5M #1.5M
                "run_number": run_number,
                "arrival_rate": exp_args["arrival-rate"],
                "arrival_seed": 100234 + i * 100101 + j * 10223,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "quantile_key": bench_params[key_this_run],  # quantile key
                "until": int(
                    exp_args["until"]  # 00
                ),  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
                "module_label": exp_args["module_label"],
                "predictor_type": exp_args["predictor_type"],
            }

            # complete parameters from the module
            # call by reference
            set_param_func = exp_args["module_set_param"]
            set_param_func(params)

            return_dict[run_number] = manager.dict()
            p = mp.Process(
                target=process,
                args=(
                    params,
                    return_dict,
                    exp_args["module_run"],
                    exp_args["module_label"],
                ),
            )
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
                exit(0)

    # save the json info
    # {"arrival_rate": 0.095, "arrival_label": "highutil", "utilization":0.9669104987005441}
    res_dict = return_dict[0].copy()
    jsoninfo = json.dumps(
        {
            "arrival_rate": exp_args["arrival-rate"],
            "label": exp_args["label"],
            "utilization": res_dict["utilization"],
        }
    )
    with open(project_path + "info.json", "w") as f:
        f.write(jsoninfo)

    for run_number in return_dict:
        print(return_dict[run_number].copy())


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "run":
        exp_args = parse_run_args(argv[1:])
        run_processes(exp_args)
    elif argv[0] == "gym":
        gym_args = parse_gym_args(argv[1:])
        run_gym_processes(gym_args)
    elif argv[0] == "train":
        train_args = parse_train_args(argv[1:])
        run_train_processes(train_args)
    elif argv[0] == "plot":
        plot_args = parse_plot_args(argv[1:])
        plot_main(plot_args)
    else:
        raise Exception("wrong command line option")
