import json
import os
import time
from typing import Tuple

import numpy as np
import polars as pl
from loguru import logger


def run_core(params, queue) -> Tuple[pl.DataFrame, dict]:

    results = {}

    # Must move all tf context initializations inside the child process
    from qsimpy.core import Model, Source, TimedSource
    from qsimpy.polar import PolarSink
    from qsimpy.random import Deterministic

    module_label = params["module_label"]

    records_path = params["records_path"] + module_label + "/"
    os.makedirs(records_path, exist_ok=True)

    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"{module_label} benchmark #{params['run_number']}")

    # Create a source
    # arrival process deterministic
    arrival = Deterministic(
        rate=params["arrival_rate"],
        seed=params["arrival_seed"],
        dtype="float64",
    )
    if module_label == "noaqm":
        source = Source(
            name="start-node",
            arrival_rp=arrival,
            task_type="0",
        )
    else:
        source = TimedSource(
            name="start-node",
            arrival_rp=arrival,
            task_type="0",
            delay_bound=params["delay_bound"],
        )
    model.add_entity(source)

    # Queue and Server
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
        records_path + f"{params['run_number']}_{module_label}_model.json",
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
    results["utilization"] = service_mean * arrival.rate

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
        "{0}: {1}: Run finished in {2} seconds".format(
            params["run_number"], module_label, end - start
        )
    )

    logger.info(
        "{0}: {1}: Source generated {2} tasks".format(
            params["run_number"], module_label, source.get_attribute("tasks_generated")
        )
    )
    logger.info(
        "{0}: {1}: Queue completed {2}, dropped {3}".format(
            params["run_number"],
            module_label,
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: {1}: Sink received {2} tasks".format(
            params["run_number"], module_label, sink.get_attribute("tasks_received")
        )
    )

    # Process the collected data
    df = sink.received_tasks
    # print(df)

    df.write_parquet(
        file=records_path + f"{params['run_number']}_{module_label}_records.parquet",
        compression="snappy",
    )

    dropped_df = df.filter(pl.col("end_time") == -1)
    passed_df = df.filter(pl.col("end_time") != -1)
    if module_label == "noaqm":
        delayed_df = passed_df.filter(False)  # give an empty dataframe
    else:
        delayed_df = passed_df.filter(pl.col("end2end_delay") > pl.col("delay_bound"))

    failed_ratio = (dropped_df.height + delayed_df.height) / df.height
    logger.info(
        f"{params['run_number']}: {module_label}: total={df.height}, "
        + f"passed={passed_df.height}, dropped={dropped_df.height} "
        + f"delayed={delayed_df.height}, failed ratio={failed_ratio} ",
    )

    results["total"] = df.height
    results["passed"] = passed_df.height
    results["dropped"] = dropped_df.height
    results["delayed"] = delayed_df.height
    results["failed_ratio"] = failed_ratio

    resultjson = json.dumps(results.copy())
    with open(
        records_path + f"{params['run_number']}_{module_label}_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(resultjson)

    return df, results
