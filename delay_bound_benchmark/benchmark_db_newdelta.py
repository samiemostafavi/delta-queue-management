import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from loguru import logger

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_all(params, return_dict, main_benchmark: Callable, main_benchmark_name: str):

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


def run_noaqm(params, return_dict):

    from qsimpy.core import Model, Source
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic
    from qsimpy.simplequeue import SimpleQueue
    from qsimpy_aqm.random import HeavyTailGamma

    records_path = params["records_path"] + "noaqm/"
    os.makedirs(records_path, exist_ok=True)

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"NOAQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=params["arrival_rate"],
        seed=params["arrival_seed"],
        dtype="float64",
    )
    source = Source(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
    )
    model.add_entity(source)

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.1,
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    queue = SimpleQueue(
        name="queue",
        service_rp=service,
        # queue_limit=10, #None
    )
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = PolarSink(
        name="sink",
        batch_size=10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

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
                    "service_end": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        records_path + f"{params['run_number']}_noaqm_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    service_mean = np.mean(queue.service_rp._pregenerated_samples)
    logger.info(
        f"Service mean: {service_mean}, "
        + f"arrival mean: {1.00/arrival.rate}, utilization: {service_mean*arrival.rate}"
    )
    return_dict[params["run_number"]]["utilization"] = service_mean * arrival.rate

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"NOAQM {params['run_number']}: simulation progress "
            + f"{100.0*float(model.env.now)/float(params['until'])}% done",
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(
        "{0}: Run finished in {1} seconds".format(params["run_number"], end - start)
    )

    logger.info(
        "{0}: Source generated {1} tasks".format(
            params["run_number"], source.get_attribute("tasks_generated")
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
        "{0}: Sink received {1} tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    # Process the collected data
    df = sink.received_tasks
    # print(df)

    df.write_parquet(
        file=records_path + f"{params['run_number']}_noaqm_records.parquet",
        compression="snappy",
    )

    df = df.select([pl.col("end2end_delay")])
    res = df.quantile(
        quantile=params["quantile_key"],
        interpolation="nearest",
    )
    return_dict[params["run_number"]]["quantile_key"] = params["quantile_key"]
    return_dict[params["run_number"]]["quantile_value"] = res[0, 0]
    return_dict[params["run_number"]]["until"] = params["until"]

    resultjson = json.dumps(return_dict[params["run_number"]].copy())
    with open(
        records_path + f"{params['run_number']}_noaqm_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)


def run_newdelta(params, return_dict):

    from qsimpy.core import Model, TimedSource
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic
    from qsimpy.simplequeue import SimpleQueue
    from qsimpy_aqm.delta import PredictorAddresses
    from qsimpy_aqm.newdelta import Horizon, NewDeltaQueue
    from qsimpy_aqm.random import HeavyTailGamma

    records_path = params["records_path"] + "newdelta/"
    os.makedirs(records_path, exist_ok=True)

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"New Delta AQM benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=params["arrival_rate"],  # 0.09 low or 0.095 high
        seed=params["arrival_seed"],
        dtype="float64",
    )
    delay_bount = return_dict[params["run_number"]]["quantile_value"]
    source = TimedSource(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
        delay_bound=delay_bount,
    )
    model.add_entity(source)

    # Queue and Server
    # service process a HeavyTailGamma
    service = HeavyTailGamma(
        seed=params["service_seed"],
        gamma_concentration=5,
        gamma_rate=0.5,
        gpd_concentration=0.1,
        threshold_qnt=0.8,
        dtype="float64",
        batch_size=params["arrivals_number"],
    )
    queue = NewDeltaQueue(
        name="queue",
        service_rp=service,
        predictor_addresses=PredictorAddresses(
            h5_address=params["predictor_addr_h5"],
            json_address=params["predictor_addr_json"],
        ),
        horizon=Horizon(
            max_length=10,
            min_length=None,
            arrival_rate=None,
        ),
        limit_drops=[0, 1, 2, 3],
        gradient_check=True,
        debug_drops=False,
        do_not_drop=False,
    )
    model.add_entity(queue)

    # Sink: to capture both finished tasks and dropped tasks (PolarSink to be faster)
    sink = PolarSink(
        name="sink",
        batch_size=10000,
    )
    # define postprocess function: the name must be 'user_fn'

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["service_delay"] = df["end_time"] - df["service_time"]
        df["queue_delay"] = df["service_time"] - df["queue_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # Wire start-node, queue, end-node, and sink together
    source.out = queue.name
    queue.out = sink.name
    queue.drop = sink.name

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
                    "service_end": "end_time",
                },
            },
            "attributes": {
                source.name: {
                    "task_generation": {
                        queue.name: {
                            "queue_length": "queue_length",
                        },
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        records_path + f"{params['run_number']}_newdelta_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    service_mean = np.mean(queue.service_rp._pregenerated_samples)
    logger.info(
        f"Service mean: {service_mean}, "
        + f"arrival mean: {1.00/arrival.rate}, utilization: {service_mean*arrival.rate}"
    )
    return_dict[params["run_number"]]["utilization"] = service_mean * arrival.rate

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"{params['run_number']}: Simulation progress "
            + f"{100.0*float(model.env.now)/float(params['until'])}% done"
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(
        "{0}: NEWDELTA: Run finished in {1} seconds".format(
            params["run_number"], end - start
        )
    )

    logger.info(
        "{0}: NEWDELTA: Source generated {1} tasks".format(
            params["run_number"], source.get_attribute("tasks_generated")
        )
    )
    logger.info(
        "{0}: NEWDELTA: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: NEWDELTA: Sink received {1} tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    # Process the collected data
    df = sink.received_tasks
    # print(df)

    df.write_parquet(
        file=records_path + f"{params['run_number']}_newdelta_records.parquet",
        compression="snappy",
    )

    dropped_df = df.filter(pl.col("end_time") == -1)
    passed_df = df.filter(pl.col("end_time") != -1)
    delayed_df = passed_df.filter(pl.col("end2end_delay") > pl.col("delay_bound"))

    failed_ratio = (dropped_df.height + delayed_df.height) / df.height
    logger.info(
        f"{params['run_number']}: NEWDELTA: total={df.height}, "
        + f"passed={passed_df.height}, dropped={dropped_df.height} "
        + f"delayed={delayed_df.height}, failed ratio={failed_ratio} ",
    )

    return_dict[params["run_number"]]["total"] = df.height
    return_dict[params["run_number"]]["passed"] = passed_df.height
    return_dict[params["run_number"]]["dropped"] = dropped_df.height
    return_dict[params["run_number"]]["delayed"] = delayed_df.height
    return_dict[params["run_number"]]["failed_ratio"] = failed_ratio

    resultjson = json.dumps(return_dict[params["run_number"]].copy())
    with open(
        records_path + f"{params['run_number']}_newdelta_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)


if __name__ == "__main__":

    # arrival_rate = {"value": 0.09, "name": "lowutil"}
    arrival_rate = {"value": 0.095, "name": "highutil"}

    # arrival_rate = {
    #    'value' : 0.095,
    #    'name' : 'highutil'
    # }

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + "/" + arrival_rate["name"] + "_results/"
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

            run_number = j * parallel_runs + i
            params = {
                "run_noaqm": True,
                "records_path": records_path,
                "arrivals_number": 1000000,  # 5M #1.5M
                "run_number": run_number,
                "arrival_rate": arrival_rate["value"],
                "arrival_seed": 100234 + i * 100101 + j * 10223,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "quantile_key": bench_params[key_this_run],  # quantile key
                "until": int(
                    1000000  # 00
                ),  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # 0.05 # report when 10%, 20%, etc progress reaches
                "predictor_addr_h5": predictors_path + "gmevm_model.h5",
                "predictor_addr_json": predictors_path + "gmevm_model.json",
            }
            return_dict[run_number] = manager.dict()
            p = mp.Process(
                target=run_all,
                args=(
                    params,
                    return_dict,
                    run_newdelta,
                    "Newdelta",
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

    for run_number in return_dict:
        print(return_dict[run_number].copy())
