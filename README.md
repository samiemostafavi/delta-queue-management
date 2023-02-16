[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# delta-queue-management

To reproduce the simulations and results of the paper "Delta: Predictive Queue Management for Time Sensitive Stochastic Networking"

## Start

Create a Python 3.9 virtual environment using `virtualenv`

        $ python -m virtualenv --python=python3.9 ./venv
        $ source venv/bin/activate

Install dependencies

        $ pip install numpy==1.23.3
        $ pip install pandas==1.5.0
        $ pip install -Ur requirements.txt

To use pyspark, install Java on your machine

        $ sudo apt-get install openjdk-8-jdk

To use SciencePlot, latex must be installed on your machine

        $ sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super

## Benchmarking Delta against other AQM schemes

We assume that an ideal predictor exists for Delta to utilize. It is located in `./predictors` folder.

Tune CoDel, note that `until` argument above 100k makes the run much longer
```
python -m tune_codel -a 0.09 -u 100000 -l tune_lowutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm
python -m tune_codel -a 0.095 -u 100000 -l tune_highutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm
```

Train Deep Queue Network implementation as follows. We set the `delay_ref` parameter to the target delay obtained by tuning CoDel.
```
python -m train_deepq -a 0.09 -u 100000 -e 1000000 -l lowutil --interval 2 --delta 0.8 --run-noaqm
python -m train_deepq -a 0.095 -u 100000 -e 1000000 -l highutil --interval 2 --delta 0.9 --run-noaqm
```

Run benchmarks

With lower utilization:
```
python -m otherschemes_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module delta --run-noaqm
python -m otherschemes_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module offline-optimum
python -m otherschemes_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module codel
python -m otherschemes_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module deepq
```

With higher utilization:
```
python -m otherschemes_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module delta --run-noaqm
python -m otherschemes_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module offline-optimum
python -m otherschemes_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module codel
python -m otherschemes_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module deepq
```

Plot the benchmark results
```
python -m otherschemes_benchmark plot --project lowutil --models deepq,codel,delta,offline-optimum --type png
python -m otherschemes_benchmark plot --project highutil --models deepq,codel,delta,offline-optimum --type png
```

## Sensitivity analysis of Delta AQM

### Number of samples in predictor training

Create a dataset for q states from 0 to 14, each 200000 samples. GPD concentration is 0.3 (p3).
```
python -m numsamples_benchmark gym -s 200000 -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -l gym_p3 -g 0.3 -r 30
python -m numsamples_benchmark gym -s 200000 -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -l gym_p0 -g 0.001 -r 30
```

Train 9 predictors with 64 samples, on the p3 dataset then validate the first predictor (`-e 0`):
```
python -m numsamples_benchmark train -d gym_p3 -l train_p3_64 -c numsamples_benchmark/train_conf_64_nogmm.json -e 9
python -m numsamples_benchmark validate -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -d gym_p3 -m train_p3_16.gmevm -l validate_p3_64 -r 3 -c 5 -y 0,100,250 -e 0
```

Train 9 predictors with 64 samples, on the p0 dataset then validate the first predictor (`-e 0`):
```
python -m numsamples_benchmark train -d gym_p0 -l train_p0_64 -c numsamples_benchmark/train_conf_64_nogmm.json -e 9
python -m numsamples_benchmark validate -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -d gym_p3 -m train_p3_16.gmevm -l validate_p3_64 -r 3 -c 5 -y 0,100,250 -e 0
```


Run the queueing simulation with Delta aqm, no aqm, and offline-optimum aqm
CAUTION: delta benchmarks with the following configuration (1M seconds) on a system with 18 cores take about 8 hours. Offline-optimum takes half an hour.
```
python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_32768 -m train_p3_32768.gmevm -r 4 -g 0.3 -e 9 --run-noaqm

python -m numsamples_benchmark run -a 0.09 -u 10000000 -l run_p0_32768 -m train_p0_32768.gmevm -r 4 -g 0.001 -e 8 --run-noaqm

python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_4096 -m train_p3_4096.gmevm -r 4 -g 0.3 -e 9 --run-noaqm
python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_4096 -m offline-optimum -r 4 -g 0.3 -e 9 --run-noaqm

python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_128 -m train_p3_128.gmevm -r 4 -g 0.3 -e 9 --run-noaqm
python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_128 -m offline-optimum -r 4 -g 0.3 -e 9 --run-noaqm

python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_64 -m train_p3_64.gmevm -r 4 -g 0.3 -e 9 --run-noaqm
python -m numsamples_benchmark run -a 0.088 -u 1000000 -l run_p3_64 -m offline-optimum -r 4 -g 0.3 -e 9 --run-noaqm
```


Plot the results
```
python -m numsamples_benchmark plot --project run_p0_32768 --models noaqm,gmevm,oo --type png

python -m numsamples_benchmark plot --project run_p3_32768 --models noaqm,gmevm,oo --type png
python -m numsamples_benchmark plot --project run_p3_4096 --models noaqm,gmevm,oo --type png
python -m numsamples_benchmark plot --project run_p3_128 --models noaqm,gmevm,oo --type png
python -m numsamples_benchmark plot --project run_p3_64 --models noaqm,gmevm,oo --type png

python -m numsamples_benchmark plot --project agg --models noaqm,gmevm_64,gmevm_128,gmevm_4096,gmevm_32k,oo --type png
```


## Multi-hop Benchmarks

This is an example for a two hop case. 

Produce data for training delay predictors and validate it. We produce samples for 2 hop and 1 hop cases since for a 2 hop network, we need both.

```
python -m multihop_benchmark gym -s 10000 -q 0,2,4,6,8,10 -d 0.1,0.5,0.9 -p 2 -l gym_2hop_p2 -g 0.2
python -m multihop_benchmark validate_gym -q [0,2],[4,4],[6,6],[0,10],[8,10] -d gym_2hop_p2 -w [0.1,0.1],[0.1,0.9],[0.9,0.9] -l validate_gym_2hop_p2 -r 3 -c 5 -y 0,100,250
```

```
python -m multihop_benchmark gym -s 10000 -q 0,2,4,6,8,10 -d 0.1,0.5,0.9 -p 1 -l gym_1hop_p2 -g 0.2
python -m multihop_benchmark validate_gym -q [0],[4],[6],[8],[10] -d gym_1hop_p2 -w [0.1],[0.5],[0.9] -l validate_gym_1hop_p2 -r 3 -c 5 -y 0,100,250
```

Train predictors

```
python -m multihop_benchmark train -d gym_2hop_p2 -l train_2hop_p2 -c multihop_benchmark/train_conf_2hop.json -e 10
python -m multihop_benchmark validate_pred -q [0,2],[4,4],[6,6],[0,10],[8,10] -d gym_2hop_p2 -w [0.1,0.1],[0.1,0.9],[0.9,0.9] -m train_2hop_p2.gmm,train_2hop_p2.gmevm -l validate_pred_2hop_p2 -r 3 -c 5 -y 0,100,250 -e 1
```

```
python -m multihop_benchmark train -d gym_1hop_p2 -l train_1hop_p2 -c multihop_benchmark/train_conf_1hop.json -e 10
python -m multihop_benchmark validate_pred -q [0],[4],[6],[8],[10] -d gym_1hop_p2 -w [0.1],[0.5],[0.9] -m train_1hop_p2.gmm,train_1hop_p2.gmevm -l validate_pred_1hop_p2 -r 3 -c 5 -y 0,100,250 -e 1
```
        
# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files

