[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# delta-queue-management
To reproduce the simulations and results of the paper "Delta: Predictive Queue Management for Time Sensitive Stochastic Networking"

# Start

Create a Python 3.9 virtual environment using `virtualenv`

        $ python -m virtualenv --python=python3.9 ./venv
        $ source venv/bin/activate

Install dependencies

        $ pip install -Ur requirements.txt

# Usage

Tune CoDel, note that `until` argument above 100k makes the run much longer

        $ python -m tune_codel -a 0.09 -u 100000 -l tune_lowutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm
        $ python -m tune_codel -a 0.095 -u 100000 -l tune_highutil --target-bounds 1,300 --target-initial 20 --interval-bounds 1,300 --interval-initial 10 --run-noaqm

Train Deep Queue Network implementation as follows. We set the `delay_ref` parameter to the target delay obtained by tuning CoDel.

        $ python -m train_deepq -a 0.09 -u 100000 -e 1000000 -l lowutil --interval 2 --delta 0.8 --run-noaqm
        $ python -m train_deepq -a 0.095 -u 100000 -e 1000000 -l highutil --interval 2 --delta 0.9 --run-noaqm

Run delay bound benchmarks

        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module offline-optimum
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module codel
        $ python -m delay_bound_benchmark run --arrival-rate 0.09 --until 1000000 --label lowutil --module deepq
        
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module delta --run-noaqm
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module offline-optimum
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module codel
        $ python -m delay_bound_benchmark run --arrival-rate 0.095 --until 1000000 --label highutil --module deepq

Plot the delay bound benchmark results

        $ python -m delay_bound_benchmark plot --project lowutil --models deepq,codel,delta,offline-optimum --type png
        $ python -m delay_bound_benchmark plot --project highutil --models deepq,codel,delta,offline-optimum --type png

Train predictors with different number of samples and models, then validate them

        $ python -m models_benchmark gym -s 10000 -q 0,3,6,9,12,15,18,21,24 -l gym_p2short -g 0.2 -r 20
        $ python -m models_benchmark gym -s 10000 -q 0,10,20,30,40,50 -l gym_p3long -g 0.3 -r 20

        $ python -m models_benchmark train -d gym_p3long -l train_p3long -c models_benchmark/train_conf_lowsample.json
        $ python -m models_benchmark validate -q 0,10,20,30,40,50 -d gym_p3long -m train_p3long.gmm,train_p3long.gmevm -l valide_p3long -r 2 -c 3 -y 0,100,800

        $ python -m models_benchmark train -d gym_p3 -l train_p3_512 -c models_benchmark/train_conf_512.json -e 10
        $ python -m models_benchmark validate -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -d gym_p3 -m train_p3_512.gmm,train_p3_512.gmevm -l validate_p3_512 -r 3 -c 5 -y 0,100,250 -e 1

        $ python -m models_benchmark train -d gym_p3 -l train_p3_128 -c models_benchmark/train_conf_128.json -e 10
        $ python -m models_benchmark validate -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -d gym_p3 -m train_p3_128.gmm,train_p3_128.gmevm -l validate_p3_128 -r 3 -c 5 -y 0,100,250 -e 0

Run delta models benchmarks
        $ python -m models_benchmark run -a 0.089 -u 1000000 -l lowutil_p3 -m train_p3_128.gmevm.0 -r 4 -g 0.3 --run-noaqm


        $ python -m models_benchmark run -a 0.089 -u 1000000 -l lowutil -m gmevm --run-noaqm
        $ python -m models_benchmark run -a 0.089 -u 1000000 -l lowutil -m gmm --run-noaqm

        $ python -m models_benchmark run -a 0.094 -u 1000000 -l highutil -m gmevm --run-noaqm
        $ python -m models_benchmark run -a 0.094 -u 1000000 -l highutil -m gmm --run-noaqm

Plot delta models benchmark results

        $ python -m delta_models_benchmark plot --project lowutil --models gmm,gmevm,offline-optimum --type png
        $ python -m delta_models_benchmark plot --project highutil --models gmm,gmevm,offline-optimum --type png

# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files


# Demo commands to run:
        1. python -m models_benchmark gym -s 10000 -q 0,3,6,9,12,15,18,21,24 -l gym_p2short -g 0.6 -r 20
        2. python -m models_benchmark train -d gym_p2short -l train_p2short -c models_benchmark/train_conf.json -e 1
        3. python -m models_benchmark validate -q 0,3,6,9,12,15,18,21,24 -d gym_p2short -m train_p2short.gmm,train_p2short.gmevm -l validate_p2short -r 3 -c 5 -y 0,100,250 -e 0
        4. python -m models_benchmark run -a 0.089 -u 10000 -l lowutil_p2short -m offline-optimum.oo -r 4 -g 0.6 --run-noaqm
        5. python -m models_benchmark plot --project lowutil_p2short --models gmm,gmevm,oo --type png


 # g specific commands to run
        1. python -m models_benchmark gym -s 10000 -q 0,3,6,9,12,15,18,21,24 -l gym_p2shortg5 -g 0.5 -r 18
        2. python -m models_benchmark train -d gym_p2shortg5 -l train_p2shortg5 -c models_benchmark/train_conf.json -e 1
        3. python -m models_benchmark validate -q 0,3,6,9,12,15,18,21,24 -d gym_p2shortg5 -m train_p2shortg5.gmm,train_p2shortg5.gmevm -l validate_p2shortg5 -r 3 -c 5 -y 0,100,250 -e 0
        4. python -m models_benchmark run -a 0.0869 -u 100000 -l lowutil_p2shortg5 -m offline-optimum.oo -r 16 -g 0.5 --run-noaqm
        5. python -m models_benchmark run -a 0.0869 -u 5000 -l lowutil_p2shortg5 -m train_p2shortg5.gmevm.0 -r 4 -g 0.5 --run-noaqm
        6. python -m models_benchmark run -a 0.0869 -u 50000 -l lowutil_p2shortg5 -m train_p2shortg5.gmm.0 -r 4 -g 0.5 --run-noaqm
        7. python -m models_benchmark plot --project lowutil_p2shortg5 --models gmm,gmevm,oo --type png

