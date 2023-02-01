import getopt
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from .aqm import run_aqm
from .gym import parse_gym_args, run_gym_processes
from .noaqm import run_noaqm
from .plot import parse_plot_args, plot_main
from .train import parse_train_args, run_train_processes
from .validate import parse_validate_args, run_validate_processes

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_run_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    args_dict["run-noaqm"] = False
    try:
        opts, args = getopt.getopt(
            argv,
            "ha:u:l:m:r:e:g:n",
            [
                "arrival-rate=",
                "until=",
                "label=",
                "model=",
                "runs=",
                "ensembles=",
                "gpd_concentration=",
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
                + "-a <arrival rate> -u <until> -l <label> -m <model> -r <number of runs> -g <gpd concentration>",
            )
            sys.exit()
        elif opt in ("-a", "--arrival-rate"):
            args_dict["arrival-rate"] = float(arg)
        elif opt in ("-u", "--until"):
            args_dict["until"] = int(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-m", "--model"):
            # <train-label>.<model-type>
            # e.g. train_p3_128.gmm
            args_dict["model"] = [s.strip() for s in arg.split(".")]
            if args_dict["model"][0] == "offline-optimum":
                args_dict["module_label"] = "oo"
            else:
                args_dict["module_label"] = args_dict["model"][1]
        elif opt in ("-r", "--runs"):
            args_dict["n_runs"] = int(arg)
        elif opt in ("-e", "--ensembles"):
            args_dict["n_ensembles"] = int(arg)
        elif opt in ("-g", "--gpd-concentration"):
            args_dict["gpd_concentration"] = float(arg)
        elif opt in ("-n", "--run-noaqm"):
            args_dict["run-noaqm"] = True

    return args_dict


def run_processes(exp_args: dict):
    # this function creates the param dict for the run

    logger.info(
        "Prepare models benchmark experiment args "
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

    n_workers = 18
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # create params list for each run
    n_runs = exp_args["n_runs"] * exp_args["n_ensembles"]
    params_list = []
    for run_number in range(n_runs):

        # parameter figure out
        keys = list(bench_params.keys())
        quotient, remainder = divmod(run_number, len(keys))
        key_this_run = keys[remainder]
        ensemble_number = quotient % exp_args["n_ensembles"]

        # model figure out
        if exp_args["model"][0] != "offline-optimum":
            # predictor address setting
            predictor_path_h5 = (
                str(p)
                + "/"
                + exp_args["model"][0]
                + "_results/"
                + exp_args["model"][1]
                + "/model_"
                + str(ensemble_number)
                + ".h5"
            )
            predictor_path_json = (
                str(p)
                + "/"
                + exp_args["model"][0]
                + "_results/"
                + exp_args["model"][1]
                + "/model_"
                + str(ensemble_number)
                + ".json"
            )
        else:
            predictor_path_h5 = None
            predictor_path_json = None

        # create and prepare the results directory
        records_path = project_path + key_this_run + "/"
        os.makedirs(records_path, exist_ok=True)

        # save the json info
        jsoninfo = json.dumps({"quantile_key": bench_params[key_this_run]})
        with open(records_path + "info.json", "w") as f:
            f.write(jsoninfo)

        params = {
            "run_number": run_number,
            "ensemble_number": ensemble_number,
            "module_label": exp_args["module_label"],
            "module": exp_args["model"],
            "main_path": main_path,
            "project_path": project_path,
            "records_path": records_path,
            "predictor_path_h5": predictor_path_h5,
            "predictor_path_json": predictor_path_json,
            "run_noaqm": exp_args["run-noaqm"],
            "arrivals_number": 1000000,  # 5M #1.5M
            "arrival_rate": exp_args["arrival-rate"],
            "arrival_seed": 100234 + run_number * 100101 + ensemble_number * 98700,
            "service_seed": 120034 + run_number * 200202 + ensemble_number * 11992,
            "gpd_concentration": exp_args["gpd_concentration"],
            "quantile_key": bench_params[key_this_run],  # quantile key
            "until": int(
                exp_args["until"]  # 00
            ),  # 10M timesteps takes 1000 seconds, generates 900k samples
            "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
        }
        params_list.append(params)

    # start noaqm runs
    try:
        logger.info(f"Starting {n_runs} noaqm jobs")
        res = pool.map_async(run_noaqm, params_list)
        logger.info("Waiting for results")
        noaqm_results = (
            res.get()
        )  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal noaqm termination")
        pool.close()

    # process noaqm results
    for run_number in range(n_runs):
        params_list[run_number]["delay_bound"] = noaqm_results[run_number][
            "quantile_value"
        ]
        logger.info(
            f"Run number {run_number}: noaqm results {noaqm_results[run_number]}"
        )
        logger.info(
            f"Run number {run_number}: set delay_bound={params_list[run_number]['delay_bound']}"
        )
        measured_utilization = noaqm_results[run_number]["misc"]["utilization"]

    # save the json info
    # {"arrival_rate": 0.095, "arrival_label": "highutil", "utilization":0.9669104987005441}
    jsoninfo = json.dumps(
        {
            "arrival_rate": exp_args["arrival-rate"],
            "label": exp_args["label"],
            "utilization": measured_utilization,
        }
    )
    with open(project_path + "info.json", "w") as f:
        f.write(jsoninfo)

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # start aqm runs
    try:
        logger.info(f"Starting {n_runs} aqm jobs")
        res = pool.map_async(run_aqm, params_list)
        logger.info("Waiting for results")
        aqm_results = (
            res.get()
        )  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal noaqm termination")
        pool.close()

    # process noaqm results
    for run_number in range(n_runs):
        logger.info(f"Run number {run_number}: aqm results {aqm_results[run_number]}")


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
    elif argv[0] == "validate":
        train_args = parse_validate_args(argv[1:])
        run_validate_processes(train_args)
    elif argv[0] == "plot":
        plot_args = parse_plot_args(argv[1:])
        plot_main(plot_args)
    else:
        raise Exception("wrong command line option")
