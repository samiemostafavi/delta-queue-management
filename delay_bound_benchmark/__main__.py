import getopt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Callable

from loguru import logger

from .newdelta import run_newdelta
from .noaqm import run_noaqm
from .offlineoptimum import run_offlineoptimum
from .plot import plot_main


def main_process(
    params, return_dict, main_benchmark: Callable, main_benchmark_name: str
):

    if params["run_noaqm"]:
        logger.info(f"{params['run_number']}: Running NOAQM")
        run_noaqm(params, return_dict)
    else:
        logger.info(f"{params['run_number']}: Not running NOAQM, reading results")
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
            f"{params['run_number']}: "
            + f"quantile_value={noaqm_res['quantile_value']}, "
            + f"quantile_key={noaqm_res['quantile_key']}, "
            + f"until={noaqm_res['until']}"
        )

        return_dict[params["run_number"]]["quantile_value"] = noaqm_res[
            "quantile_value"
        ]
        return_dict[params["run_number"]]["quantile_key"] = noaqm_res["quantile_key"]
        return_dict[params["run_number"]]["until"] = noaqm_res["until"]

    logger.info(f"{params['run_number']}: Running {main_benchmark_name}")
    main_benchmark(params, return_dict)


def parse_run_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    args_dict["run-noaqm"] = False
    try:
        opts, args = getopt.getopt(
            argv,
            "ha:u:l:m:n",
            ["arrival-rate=", "until=", "label=", "module=", "run-noaqm"],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m delay_bound_benchmark -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m delay_bound_benchmark run "
                + "-a <arrival rate> -u <until> -l <label>",
            )
            sys.exit()
        elif opt in ("-a", "--arrival-rate"):
            args_dict["arrival-rate"] = float(arg)
        elif opt in ("-u", "--until"):
            args_dict["until"] = int(arg)
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-m", "--module"):
            if arg == "delta":
                args_dict["module_callable"] = run_newdelta
                args_dict["module_label"] = "delta"
            elif arg == "offline-optimum":
                args_dict["module_callable"] = run_offlineoptimum
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
        print('Wrong args, type "python -m delay_bound_benchmark -h" for help')
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


def run_main(exp_args: dict):

    logger.info(f"Running delay-bound benchmark experiment with args: {exp_args}")

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + "/" + exp_args["label"] + "_results/"
    predictors_path = str(p) + "/predictors/"
    os.makedirs(project_path, exist_ok=True)

    # simulation parameters
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
            with open(
                records_path + "info.json",
                "w",
            ) as f:
                f.write(jsoninfo)

            run_number = j * parallel_runs + i
            params = {
                "run_noaqm": exp_args["run-noaqm"],
                "records_path": records_path,
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
                "predictor_addr_h5": predictors_path + "gmevm_model.h5",
                "predictor_addr_json": predictors_path + "gmevm_model.json",
            }
            return_dict[run_number] = manager.dict()
            p = mp.Process(
                target=main_process,
                args=(
                    params,
                    return_dict,
                    exp_args["module_callable"],
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
    with open(
        project_path + "info.json",
        "w",
    ) as f:
        f.write(jsoninfo)

    for run_number in return_dict:
        print(return_dict[run_number].copy())


if __name__ == "__main__":

    argv = sys.argv[1:]
    if argv[0] == "run":
        exp_args = parse_run_args(argv[1:])
        run_main(exp_args)
    elif argv[0] == "plot":
        plot_args = parse_plot_args(argv[1:])
        plot_main(plot_args)
    else:
        raise Exception("wrong command line option")
