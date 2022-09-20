import getopt
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def parse_gym_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hs:q:l:g:",
            [
                "samples=",
                "qlens=",
                "label=",
                "gpd_concentration=",
            ],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark gym -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark gym "
                + "-s <number of samples> -q <queue lengths> -l <label> -g <gpd concentration>",
            )
            sys.exit()
        elif opt in ("-s", "--samples"):
            args_dict["samples"] = int(arg)
        elif opt in ("-q", "--qlens"):
            args_dict["qlens"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["module_label"] = arg
        elif opt in ("-g", "--gpd-concentration"):
            args_dict["gpd_concentration"] = float(arg)

    return args_dict


def run_gym_processes(exp_args: dict):
    logger.info(
        "Prepare models benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["module_label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # experiment parameters
    # quantile values of no-aqm model with p1 as gpd_concentration
    bench_params = {f"q{qlen}": qlen for qlen in exp_args["qlens"]}  # qlens

    # manager = mp.Manager()

    sequential_runs = 1  # 2  # 2  # 4
    parallel_runs = 18  # 8  # 8  # 18
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
            jsoninfo = json.dumps({"qlen": bench_params[key_this_run]})
            with open(records_path + "info.json", "w") as f:
                f.write(jsoninfo)

            run_number = j * parallel_runs + i
            params = {
                "main_path": main_path,
                "project_path": project_path,
                "records_path": records_path,
                "run_number": run_number,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "traffic_tasks": bench_params[key_this_run],  # qlen
                "samples": exp_args["samples"],
                "report_state_perc": 5,
                "gpd_concentration": exp_args["gpd_concentration"],
                "module_label": exp_args["module_label"],
                "service_batchsize": 100000,
            }

            p = mp.Process(
                target=run_gym_noaqm,
                args=(
                    params,
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


def run_gym_noaqm(params, module_label: str) -> None:
    from qsimpy.simplequeue import SimpleQueue
    from qsimpy_aqm.random import HeavyTailGamma

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=params["gpd_concentration"],
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["service_batchsize"],
    )
    queue = SimpleQueue(
        name="queue",
        service_rp=service,
        # queue_limit=10, #None
    )

    run_gym_core(params, queue, module_label)


def run_gym_core(params, queue, module_label: str) -> None:

    # Must move all tf context initializations inside the child process
    from qsimpy.core import Model, Sink
    from qsimpy.gym import GymSink, GymSource

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym benchmark #{params['run_number']}")

    # create the gym source
    source = GymSource(
        name="start-node",
        main_task_type="main",
        traffic_task_type="traffic",
        traffic_task_num=params["traffic_tasks"],
    )
    model.add_entity(source)

    # Queue and Server
    model.add_entity(queue)

    # create the sinks
    sink = GymSink(
        name="gym-sink",
        batch_size=10000,
    )

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    drop_sink = Sink(
        name="drop-sink",
    )
    model.add_entity(drop_sink)

    # make the connections
    source.out = queue.name
    queue.out = sink.name
    queue.drop = drop_sink.name
    sink.out = source.name
    # queue should not drop any task

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                source.name: {
                    "task_generation": "start_time",
                },
                queue.name: {
                    "task_reception": "queue_time",
                    "service_start": "service_time",
                },
                sink.name: {
                    "task_reception": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                            "is_busy": "queue_is_busy",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        params["records_path"] + f"{params['run_number']}_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    def until_proc(env, step, report_samples_perc):
        old_perc = 0
        while True:
            yield env.timeout(step)
            progress_in_perc = int(
                float(sink.attributes["tasks_received"])
                / float(params["samples"])
                * 100
            )
            if (progress_in_perc % report_samples_perc == 0) and (
                old_perc != progress_in_perc
            ):
                logger.info(
                    f"{params['run_number']}:"
                    + " Simulation progress"
                    + f" {progress_in_perc}% done"
                )
                old_perc = progress_in_perc
            if sink.attributes["tasks_received"] >= params["samples"]:
                break
        return True

    # Run!
    start = time.time()
    model.env.run(
        until=model.env.process(
            until_proc(
                model.env,
                step=10,
                report_samples_perc=params["report_state_perc"],
            )
        )
    )
    end = time.time()
    logger.info(f"{params['run_number']}: Run finished in {end - start} seconds")

    logger.info(
        "{0}: Source generated {1} main tasks".format(
            params["run_number"], source.get_attribute("main_tasks_generated")
        )
    )
    logger.info(
        "{0}: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: Sink received {1} main tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    # Process the collected data
    df = sink.received_tasks

    df.write_parquet(
        file=params["records_path"] + f"{params['run_number']}_records.parquet",
        compression="snappy",
    )
