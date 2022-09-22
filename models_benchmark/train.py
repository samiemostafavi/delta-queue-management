import getopt
import json
import multiprocessing as mp
import multiprocessing.context as ctx
import os
import signal
import sys
import warnings
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore")
# very important line to make tensorflow run in sub processes
ctx._force_start_method("spawn")
# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_train_args(argv: list[str]):

    # parse arguments to a dict
    args_dict = {}
    try:
        opts, args = getopt.getopt(
            argv,
            "hd:l:c:",
            ["dataset=", "label=", "config="],
        )
    except getopt.GetoptError:
        print('Wrong args, type "python -m models_benchmark train -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                "python -m models_benchmark train "
                + "-d <dataset label> -l <label> -c <config json file>",
            )
            sys.exit()
        elif opt in ("-d", "--dataset"):
            args_dict["dataset"] = arg
        elif opt in ("-l", "--label"):
            args_dict["label"] = arg
        elif opt in ("-c", "--config-file"):
            with open(arg) as json_file:
                data = json.load(json_file)
            args_dict["train_config"] = data

    return args_dict


def run_train_processes(exp_args: list):
    logger.info(
        "Prepare models benchmark experiment args "
        + f"with command line args: {exp_args}"
    )

    # project folder setting
    p = Path(__file__).parents[0]
    main_path = str(p) + "/"
    project_path = str(p) + "/" + exp_args["label"] + "_results/"
    os.makedirs(project_path, exist_ok=True)

    train_configs = exp_args["train_config"]

    n_workers = 18
    logger.info(f"Initializng {n_workers} workers")
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(n_workers)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # create params list for each run
    n_runs = len(train_configs.keys())
    params_list = []
    for model_conf_key in train_configs.keys():
        model_conf = train_configs[model_conf_key]

        # create and prepare the results directory
        records_path = project_path + model_conf_key + "/"
        os.makedirs(records_path, exist_ok=True)

        # save the json info
        jsoninfo = json.dumps(model_conf)
        with open(records_path + "info.json", "w") as f:
            f.write(jsoninfo)

        dataset_path = main_path + exp_args["dataset"] + "_results/"
        params = {
            "dataset_path": dataset_path,
            "records_path": records_path,
            "model_conf": model_conf,
            "model_conf_key": model_conf_key,
        }
        params_list.append(params)

    try:
        logger.info(f"Starting {n_runs} jobs")
        res = pool.map_async(train_model, params_list)
        logger.info("Waiting for results")
        res.get(100)  # Without the timeout this blocking call ignores all signals.
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
    else:
        logger.info("Normal termination")
        pool.close()


def train_model(params):

    import tensorflow as tf
    from petastorm import TransformSpec
    from petastorm.spark import SparkDatasetConverter, make_spark_converter
    from pr3d.de import ConditionalGammaMixtureEVM, ConditionalGaussianMM
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import rand

    # set params
    records_folder = params["dataset_path"]
    model_conf = params["model_conf"]
    module_label = params["model_conf_key"]
    predictors_path = params["records_path"]

    # init Spark
    spark = (
        SparkSession.builder.appName("Training")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )
    # sc = spark.sparkContext

    # Set a cache directory on DBFS FUSE for intermediate data.
    file_path = dirname(abspath(__file__))
    spark_cash_addr = "file://" + file_path + "/__sparkcache_" + module_label + "__"
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, spark_cash_addr)
    logger.info(f"{module_label}: Spark cache folder is set up at: {spark_cash_addr}")

    # set data types
    npdtype = np.float64
    # tfdtype = tf.float64
    strdtype = "float64"

    # find records_paths
    records_paths = [
        records_folder + name
        for name in os.listdir(records_folder)
        if os.path.isdir(os.path.join(records_folder, name))
    ]

    logger.info(f"Opening predictors directory '{predictors_path}'")
    os.makedirs(predictors_path, exist_ok=True)

    # read all the files from the project
    files = []
    for records_path in records_paths:
        logger.info(f"Opening the path '{records_path}'")
        all_files = os.listdir(records_path)
        for f in all_files:
            if f.endswith(".parquet"):
                files.append(records_path + "/" + f)

    # read all files into one Spark df
    main_df = spark.read.parquet(*files)

    # Absolutely necessary for randomizing the rows (bug fix)
    # first shuffle, then sample!
    main_df = main_df.orderBy(rand())

    training_params = model_conf["training_params"]
    if training_params["dataset_size"] == "all":
        df_train = main_df.sample(
            withReplacement=False,
            fraction=1.00,
        )
    else:
        # take the desired number of records for learning
        df_train = main_df.sample(
            withReplacement=False,
            fraction=training_params["dataset_size"] / main_df.count(),
        )
    y_label = model_conf["y_label"]

    # dataset partitioning and making the converters
    # Make sure the number of partitions is at least the number of workers which is
    # required for distributed training.
    df_train = df_train.repartition(1)
    converter_train = make_spark_converter(df_train)
    logger.info(f"Dataset loaded, train sampels: {len(converter_train)}")

    condition_labels = model_conf["condition_labels"]
    y_label = model_conf["y_label"]

    def transform_row(pd_batch):
        """
        The input and output of this function are pandas dataframes.
        """

        pd_batch = pd_batch[[y_label, *condition_labels]]
        pd_batch["y_input"] = pd_batch[y_label]
        pd_batch = pd_batch.drop(columns=[y_label])

        # if input normalization
        pd_batch["queue_length"] = pd_batch["queue_length"]

        return pd_batch

    # Note that the output shape of the `TransformSpec` is not automatically known by petastorm,
    # so we need to specify the shape for new columns in `edit_fields` and specify the order of
    # the output columns in `selected_fields`.
    x_fields = [(cond, npdtype, (), False) for cond in condition_labels]
    transform_spec_fn = TransformSpec(
        transform_row,
        edit_fields=[
            *x_fields,
            ("y_input", npdtype, (), False),
        ],
        selected_fields=[*condition_labels, "y_input"],
    )

    model_type = model_conf["type"]
    condition_labels = model_conf["condition_labels"]
    training_rounds = training_params["rounds"]
    batch_size = training_params["batch_size"]

    # initiate the non conditional predictor
    if model_type == "gmm":
        model = ConditionalGaussianMM(
            x_dim=condition_labels,
            centers=model_conf["centers"],
            hidden_sizes=model_conf["hidden_sizes"],
            dtype=strdtype,
            bayesian=model_conf["bayesian"],
            # batch_size = 1024,
        )
    elif model_type == "gmevm":
        model = ConditionalGammaMixtureEVM(
            x_dim=condition_labels,
            centers=model_conf["centers"],
            hidden_sizes=model_conf["hidden_sizes"],
            dtype=strdtype,
            bayesian=model_conf["bayesian"],
            # batch_size = 1024,
        )

    # get into training stack
    with converter_train.make_tf_dataset(
        transform_spec=transform_spec_fn,
        batch_size=batch_size,
    ) as train_dataset:

        # tf.keras only accept tuples, not namedtuples
        # map the dataset to the desired tf.keras input in _pl_training_model
        def map_fn(x):
            x_dict = {}
            for idx, cond in enumerate(condition_labels):
                x_dict = {**x_dict, cond: x[idx]}
            return ({**x_dict, "y_input": x.y_input}, x.y_input)

        train_dataset = train_dataset.map(map_fn)

        steps_per_epoch = len(converter_train) // batch_size

        for idx, params in enumerate(training_rounds):

            logger.info(
                f"Starting training session {idx}/{len(training_rounds)} with {params}"
            )

            model._pl_training_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=params["learning_rate"]
                ),
                loss=model.loss,
            )

            logger.info(f"steps_per_epoch: {steps_per_epoch}")

            model._pl_training_model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=params["epochs"],
                verbose=1,
            )

    # training done, save the model
    model.save(predictors_path + "model.h5")
    with open(predictors_path + "model.json", "w") as write_file:
        json.dump(model_conf, write_file, indent=4)

    logger.info(
        f"A {model_type} {'bayesian' if model.bayesian else 'non-bayesian'} "
        + "model got trained and saved."
    )
