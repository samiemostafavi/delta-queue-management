# Mismatch Benchmark


1. Run the gym
- `-s`: number of samples
- `-q`: queue lengths
- `-l`: folder name to save the results
- `-c`: gamma concentration
- `-t`: gamma rate
- `-d`: gpd threshold
- `-g`: gpd concentration
- `-r`: total number of runs
- `-w`: number of workers (number of cores of the system)
```
python -m mismatch_benchmark gym -s 200000 -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -l gym_p0 -c 5 -t 0.5 -d 0.8 -g 0.001 -r 30 -w 18
```

2. Train new models
- `-d`: folder name that stores the dataset
- `-l`: folder name to save the results
- `-c`: json file containing training information
- `-e`: number of models to train
```
python -m mismatch_benchmark train -d gym_p0 -l train_p0 -c numsamples_benchmark/train_conf.json -e 9
```

3. Validate the trained models
- `-q`: queue lengths to check
- `-d`: folder name that stores the dataset
- `-m`: model to check (MODEL_FOLDER.MODEL_NAME)
- `-l`: folder name to save the results
- `-r`: number of rows in the final figure
- `-c`: number of columns in the final figure
- `-y`: y limits and resolution in the final figure (e.g. from 0 to 250, with 100 points: `0,100,250`)
- `-e`: predictor number to check
```
python -m mismatch_benchmark validate -q 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 -d gym_p0 -m train_p0.gmevm -l validate_p0 -r 3 -c 5 -y 0,100,250 -e 0
```

4. Run the queueing simulation with Delta aqm, no aqm, and offline-optimum aqm
- `-a`: arrival rate
- `-u`: until
- `-l`: folder name to save the results
- `-m`: the model to run the aqm with (MODEL_FOLDER.MODEL_NAME)
- `-c`: gamma concentration
- `-t`: gamma rate
- `-d`: gpd threshold
- `-g`: gpd concentration
- `-e`: number of predictors to run
- `-r`: total number of runs
- `-w`: number of workers (number of cores of the system)
- `--run-noaqm`: whether or not run noaqm
```
python -m mismatch_benchmark run -a 0.088 -u 1000000 -l run_p0 -m train_p0.gmevm -c 5 -t 0.5 -d 0.8 -g 0.001 -e 9 -r 4 -w 18 --run-noaqm
python -m mismatch_benchmark run -a 0.088 -u 1000000 -l run_p0 -m offline-optimum -c 5 -t 0.5 -d 0.8 -g 0.001 -e 1 -r 4
```

5. Plot the results
```
python -m mismatch_benchmark plot --project run_p0 --models noaqm,gmevm,oo --type png
```



