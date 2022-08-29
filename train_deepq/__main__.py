import getopt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

from loguru import logger

from .noaqm import run_noaqm
from .traindeepq import add_deepq_params, train_deepq


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
        return_dict[params["run_number"]]["utilization"] = noaqm_res["utilization"]
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
            "ha:u:e:l:i:d:n",
            [
                "arrival-rate=",
                "until-train=",
                "until-eval=" "label=",
                "interval=",
                "delta=",
                "run-noaqm",
            ],
        )
    except getopt.GetoptError as e:
        print(e)
        print('Wrong args, type "python -m train_deepq -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m train_dqn "
                + "-a <arrival rate> -u <until-train> "
                + "-e <until-eval> -l <label>  -i <interval> -d <delta> -r <delay ref>"
            )
            sys.exit()
        elif opt in ("-a", "--arrival-rate"):
            args_dict["arrival-rate"] = float(arg)
        elif opt in ("-u", "--until-train"):
            args_dict["until-train"] = int(arg)
        elif opt in ("-e", "--until-eval"):
            args_dict["until-eval"] = int(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-i", "--interval"):
            args_dict["interval"] = float(arg)
        elif opt in ("-d", "--delta"):
            args_dict["delta"] = float(arg)
        elif opt in ("-n", "--run-noaqm"):
            args_dict["run-noaqm"] = True

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
    parallel_runs = 4  # 8  # 8  # 18
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
                "arrivals_number": 10000,  # 5M #1.5M
                "run_number": run_number,
                "arrival_rate": exp_args["arrival-rate"],
                "arrival_seed": 100234 + i * 100101 + j * 10223,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "quantile_key": bench_params[key_this_run],  # quantile key
                "until": int(exp_args["until-train"]),
                "until_train": int(exp_args["until-train"]),  # 00
                "until_eval": int(exp_args["until-eval"]),  # 00
                "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
                "module_label": "deepq",
                "interval": exp_args["interval"],
                "delta": exp_args["delta"],
            }

            # complete parameters from the module
            # call by reference
            add_deepq_params(params)

            return_dict[run_number] = manager.dict()
            p = mp.Process(
                target=process,
                args=(
                    params,
                    return_dict,
                    train_deepq,
                    "deepq",
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

    # python -m train_deepq -a 0.09 -u 1000 -l train_lowutil --run-noaqm
    exp_args = parse_run_args(sys.argv[1:])
    run_processes(exp_args)
