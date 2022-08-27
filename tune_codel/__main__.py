import getopt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

from loguru import logger

from .noaqm import run_noaqm
from .tunecodel import tune_codel


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
            "ha:u:l:t:y:i:x:n",
            [
                "arrival-rate=",
                "until=",
                "label=",
                "target-bounds=",
                "target-initial=",
                "interval-bounds=",
                "interval-initial=",
                "run-noaqm",
            ],
        )
    except getopt.GetoptError as e:
        print(e)
        print('Wrong args, type "python -m tune_codel -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m tune_codel " + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-a", "--arrival-rate"):
            args_dict["arrival-rate"] = float(arg)
        elif opt in ("-u", "--until"):
            args_dict["until"] = int(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-t", "--target-bounds"):
            args_dict["target-bounds"] = [float(s.strip()) for s in arg.split(",")]
        elif opt in ("-y", "--target-initial"):
            args_dict["target-initial"] = float(arg)
        elif opt in ("-i", "--interval-bounds"):
            args_dict["interval-bounds"] = [float(s.strip()) for s in arg.split(",")]
        elif opt in ("-x", "--interval-initial"):
            args_dict["interval-initial"] = float(arg)
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
                "until": int(
                    exp_args["until"]  # 00
                ),  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
                "module_label": "codel",
                "interval_bounds": exp_args["interval-bounds"],
                "interval_initial": exp_args["interval-initial"],
                "target_bounds": exp_args["target-bounds"],
                "target_initial": exp_args["target-initial"],
            }

            return_dict[run_number] = manager.dict()
            p = mp.Process(
                target=process,
                args=(
                    params,
                    return_dict,
                    tune_codel,
                    "codel",
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

    # python -m tune_codel -a 0.09 -u 1000 -l tune_lowutil --target-bounds 0.1,1.0 --target-initial 0.5 --interval-bounds 0.1,2.0 --interval-initial 0.5 --run-noaqm
    exp_args = parse_run_args(sys.argv[1:])
    run_processes(exp_args)
