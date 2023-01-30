import getopt
import itertools
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_gym_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hs:q:d:p:l:g:",
            ["samples=", "qlens=", "ldps=", "hops=", "label=", "gpd_concentration="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m multihop_benchmark gym -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m multihop_benchmark gym "
                + "-s <number of samples> -q <queue lengths> -l <label> -g <gpd concentration>",
            )
            sys.exit()
        elif opt in ("-s", "--samples"):
            args_dict["samples"] = int(arg)
        elif opt in ("-q", "--qlens"):
            args_dict["qlens"] = [int(s.strip()) for s in arg.split(",")]
        elif opt in ("-l", "--label"):
            args_dict["module_label"] = arg
        elif opt in ("-p", "--hops"):
            args_dict["n_hops"] = int(arg)
        elif opt in ("-d", "--ldps"):
            args_dict["ldps"] = [float(s.strip()) for s in arg.split(",")]
        elif opt in ("-g", "--gpd-concentration"):
            args_dict["gpd_concentration"] = float(arg)

    return args_dict


def run_gym_processes(exp_args: dict):
    logger.info(
        "Prepare multihop benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["module_label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    # experiment parameters
    aq = [exp_args["qlens"] for _ in range(exp_args["n_hops"])]
    al = [exp_args["ldps"] for _ in range(exp_args["n_hops"])]
    a = [*aq, *al]

    bench_params = {}
    for idx, tup in enumerate(list(itertools.product(*a))):
        mid = {
            "qlens": [],
            "ldps": [],
        }
        for h in range(exp_args["n_hops"]):
            mid["qlens"].append(tup[h])
            mid["ldps"].append(tup[exp_args["n_hops"] + h])
        bench_params = {**bench_params, f"{idx}": mid}

    n_workers = 18
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # create params list for each run
    n_runs = len(list(bench_params.keys()))
    params_list = []
    for run_number in range(n_runs):
        key_this_run = str(run_number)

        # prepare the results directory
        records_path = project_path

        # save the json info
        jsoninfo = json.dumps(bench_params[key_this_run])
        with open(records_path + f"{key_this_run}_info.json", "w") as f:
            f.write(jsoninfo)

        params = {
            "main_path": main_path,
            "project_path": project_path,
            "records_path": records_path,
            "run_number": run_number,
            "service_seed": 120034 + run_number * 200202,
            "n_hops": exp_args["n_hops"],
            "traffic_tasks": bench_params[key_this_run]["qlens"],  # qlens
            "ldps": bench_params[key_this_run]["ldps"],  # ldps
            "samples": exp_args["samples"],
            "report_state_perc": 5,
            "gpd_concentration": exp_args["gpd_concentration"],
            "module_label": exp_args["module_label"],
            "service_batchsize": 100000,
        }
        params_list.append(params)

    try:
        logger.info(f"Starting {n_runs} jobs")
        res = pool.map_async(run_gym_noaqm, params_list)
        logger.info("Waiting for results")
        res.get(10000)  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal termination")
        pool.close()


def run_gym_noaqm(params) -> None:

    # Must move all tf context initializations inside the child process
    from qsimpy.core import Model, Sink
    from qsimpy.gym import GymSink, MultihopGymSource
    from qsimpy.simplemultihop import SimpleMultiHop
    from qsimpy_aqm.random import HeavyTailGamma

    logger.info(f"{params['run_number']}: running sub-process with params {params}")

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym benchmark #{params['run_number']}")
    logger.info(f"{params['run_number']}: starting gym benchmark")

    # create the gym source
    main_task_num = np.zeros(params["n_hops"], dtype=int)
    main_task_num[0] = 1
    source = MultihopGymSource(
        name="start-node",
        n_hops=params["n_hops"],
        main_task_type="main",
        main_task_num=list(main_task_num),
        traffic_task_type="traffic",
        traffic_task_num=list(params["traffic_tasks"]),
        traffic_task_ldp=list(params["ldps"]),
    )
    model.add_entity(source)

    # Queue and Server
    services: List[HeavyTailGamma] = []
    # service process a HeavyTailGamma
    for hop in range(params["n_hops"]):
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
            be_quiet=True,
        )
        services.append(service)

    queue = SimpleMultiHop(
        name="queue",
        n_hops=params["n_hops"],
        service_rp=services,
    )
    model.add_entity(queue)

    # create the sinks
    sink = GymSink(
        name="gym-sink",
        batch_size=10000,
    )

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]

        # process service delay, queue delay
        for h in range(params["n_hops"]):
            df[f"service_delay_{h}"] = df[f"end_time_{h}"] - df[f"service_time_{h}"]
            df[f"queue_delay_{h}"] = df[f"service_time_{h}"] - df[f"queue_time_{h}"]

        # process time-in-service
        # h is hop num
        for h in range(params["n_hops"]):
            # reduce queue_length by 1
            df[f"queue_length_h{h}"] = df[f"queue_length_h{h}"] - 1
            # process longer_delay_prob here for benchmark purposes
            df[f"longer_delay_prob_h{h}"] = source.traffic_task_ldp[h]
            del df[f"last_service_time_h{h}"], df[f"queue_is_busy_h{h}"]

        # delete remaining items
        for h in range(params["n_hops"]):
            del df[f"end_time_{h}"], df[f"service_time_{h}"], df[f"queue_time_{h}"]

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
    # timestamps
    timestamps = {
        source.name: {
            "task_generation": "start_time",
        },
        sink.name: {
            "task_reception": "end_time",
        },
        queue.name: {},
    }
    for hop in range(params["n_hops"]):
        timestamps[queue.name].update(
            {
                f"task_reception_h{hop}": f"queue_time_{hop}",
                f"service_start_h{hop}": f"service_time_{hop}",
                f"service_end_h{hop}": f"end_time_{hop}",
            }
        )
    # attributes
    attributes = {
        source.name: {
            "task_generation": {queue.name: {}},
        },
        queue.name: {
            f"service_end_h{hop}": {queue.name: {}}
            for hop in range(params["n_hops"] - 1)
        },
    }
    for hop in range(params["n_hops"]):
        attributes[source.name]["task_generation"][queue.name].update(
            {
                f"queue_length_h{hop}": f"queue_length_h{hop}",
                f"last_service_time_h{hop}": f"last_service_time_h{hop}",
                f"is_busy_h{hop}": f"queue_is_busy_h{hop}",
            }
        )

    # set attributes and timestamps
    model.set_task_records(
        {
            "timestamps": timestamps,
            "attributes": attributes,
        }
    )

    modeljson = model.json()
    with open(
        params["records_path"] + f"{params['run_number']}_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    logger.info(f"{params['run_number']}: prepare for run")

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
    logger.info(f"{params['run_number']}: start")

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
    for hop in range(params["n_hops"]):
        logger.info(
            "{0}: Queue hop {1} completed {2}, dropped {3}".format(
                params["run_number"],
                hop,
                queue.get_attribute(f"tasks_completed_h{hop}"),
                queue.get_attribute(f"tasks_dropped_h{hop}"),
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
