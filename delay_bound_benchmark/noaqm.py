import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from loguru import logger


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
