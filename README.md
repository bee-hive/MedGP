# MedGP
A distributed version of the sparse multi-output Gaussian process framework integrating python and C++.

## Installation
Download or clone the whole MedGP repository. You will need to install the python scripts and compile the C++ source files.

### (1) Install Python part of MedGP
We will need the following packages in addition to Python standard libraries:
```
numpy sklearn statsmodels matplotlib seaborn
```
To install the python packages in MedGP, please run
```
cd /your/path/to/MedGP/
python setup.py install
```
Current version has been tested under both Python2.7 and Python3.6.

### (2) Install C++ part of MedGP
First, make sure that you have intel icpc compiler installed. For example, on a linux machine, you can run
```
which icpc
```
to check the existence of the compiler with the alias icpc.

If you are using Princeton's della cluster, you can load intel compilers by
```
module load intel
```

Second, please download the [rapidjson](https://github.com/Tencent/rapidjson) library that MedGP used for configuration. After you download it, please edit the path to rapidjson library in the `Makefile`:
```
cd /your/path/to/MedGP/medgpc/src/
vim Makefile
```
and change the variable `JSON_LIB_DIR`.

Finally, edit the path to MedGP in the `Makefile`:
```
cd /your/path/to/MedGP/medgpc/src/
vim Makefile
```
and change the variable `TOP_SRC_DIR` to `/your/path/to/MedGP/medgpc/src`.

Compile through running the command in the directory where `Makefile` is:
```
make
```
, and check the existence of `main_one_train.o` and `main_one_test.o` to confirm that the compilation was successfully.

## Create your own experiments

The package includes an executable script `medgpc.util.run_exp_generator.py` that allows you to create scripts to distribute the jobs using the MedGP conveniently, as long as the data is prepared as the expected format and the system paths are configured properly. We provide an example script with the command to generate an experiment in `/your/path/to/MedGP/scripts/gen_medgpc_example.sh`. More descriptions in data preparation are given as follows:

### (1) Prepare the raw data in plain text (.txt) format
We expect you to prepare your raw data in the following hierarchy. We first describe the parameters that we will need to setup.
* `data_root_path`: the path to where the directory of the raw data is stored. You will have to assign one data set as one cohort, put them under `data_root_path/(cohort_name)/` and pass the argument `--cohort=(cohort_name)` when running `medgpc.util.run_exp_generator.py`.
* `cohort_id_list`: under `data_root_path/(cohort_name)/`, we expect you to provide a list of ids for each sample (e.g. a patient) in the format of .txt where each line provides an id. For instance,
```
(id 1)
(id 2)
...
``` 
* We expect you to put the raw time series data of each sample under `data_root_path/(cohort_name)/(sample id)/`, with each feature has its own file `feature(feature_index).txt`. `(feature_index)` is the index of the feature and should match the JSON file of features later passed to the generator (described below). The time series data per feature is presented in the format of
```
(total number of time points)
(time 1)
(value 1)
(time 2)
(value 2)
...
```
An example script of extracting the heart failure subset in MIMIC-III used in the manuscript can be found in `/scripts/jmlr_mimic_heart_failure.py`.

### (2) Prepare the configuration data in JSON (.json) format
To create the experiment, we also need four json files, passing to three arguments correspondingly: `--feature-config=(feature.json)`, `--path-config=(path.json)`, `--opt-config=(opt.json)`, `--hpc-config=(hpc.json)`. The format of each JSON file is explained as follows:

#### feature.json (example file: /scripts/feature_PT_INR.json and /scripts/feature_all.json)
This JSON file is used to specified the features to be included in the experiment, it contains a single key `feature_list`:
* `feature_list`: a **list of dictinoaries** that contains the features to be included in the experiment. The ordering of the features in the kernel will be the same as its order in the list. Each dictionary contain the keys `name` and `index` to store the name and index of the feature respectively. For instance, in `/your/path/to/MedGP/scripts/feature_PT_INR.json` we provided, there are two lab features PT and INR:
```
{
    "feature_list": [
    	{
    		"name": "INR",
    		"index": 18
    		},
    	{
    		"name": "PT",
    		"index": 19
    		}
    ]
}
```
In this case, it indicates that the time series data of INR for each subject is stored in `data_root_path/(cohort_name)/(sample id)/feature18.txt`.

#### path.json (example file: /scripts/path_della.json)
This JSON file is used to specified the path to raw data and where you want to store your experiment. It should contain the following keys:
* `medgpc_path`: the path that contains the `medgpc` package and should be the same as `/your/path/to/MedGP/` used in the Installation section above.
* `train_exec`: the name of your C++ training executable; default is `main_one_train.o` if using the default `Makefile` to compile.
* `test_exec`: the name of your C++ testing executable; default is `main_one_test.o` if using the default `Makefile` to compile.
* `exp_root_path`: the path to where the folder of the experiment will be generated under.
* `data_root_path`: the path to where all raw data are; same as described in (1) above.
* `cohort_id_list`: the list of sample ids put under `data_root_path/(cohort_name)/` with the format described in (1) above.

#### opt.json (example file: /scripts/opt_prior2.json and  /scripts/opt_prior0.json)
This JSON file is used to configure settings for optimization in MedGP training and testing. It should contain the following keys:
* `random_init_num`: the number of random initializations before starting doing optimization.
* `random_seed`: an integrer to specify the random seed used to initialize the parameters.
* `top_iteration_num`: the number of iterations for optimization. If using hierarchical gamma prior, this is the number of iterations to update the elements in A matrices. If using no prior, this is the number of iterations for scaled conjugate gradient optimizer.
* `iteration_num_per_update`: the number of iterations for scaled conjugate gradient optimizer for the inner-loop when using hierarchical gamma prior.
* `online_learn_rate`: the learning rate for online updating kernel parameters. 
* `online_momentum`: the momentum parameter for online updating kernel parameters.

In addition, the JSON file should specify the upper and lower bounds for initializing the kernel parameters. The required keys will vary by the type of kernel. As an example, when using the `LMC-SM` kernel, at least the following keys should be added:
```
"lower_bound_noise": 0.15,
"upper_bound_noise": 0.4,
"lower_bound_a": -1.5,
"upper_bound_a": 1.5,
"lower_bound_period": 12,
"upper_bound_period": 72,
"lower_bound_lengthscale": 6,
"upper_bound_lengthscale": 72,
"lower_bound_lambda": 0.1,
"upper_bound_lambda": 0.5,
"lower_bound_scale": 0.1,
"upper_bound_scale": 1.5
```

#### hpc.json (example file: /scripts/slurm_della.json)
This JSON file is used to generate scripts based on the computing cluster you are using, and provide simple setup for resource allocations. It should contain the following keys:
* `train_template`: the template for the bash executable script passing arguments to `main_one_train.o`. An example is provided in `/your/path/to/MedGP/scripts/train_della.sh`. The generator will use this template and generate headers for customized resources specified in key `train_config`; more descriptions come as below.
* `test_template`: the template for the bash executable script passing arguments to `main_one_test.o`. An example is provided in `/your/path/to/MedGP/scripts/test_della.sh`. The generator will use this template and generate headers for customized resources specified in key `test_config`; more descriptions come as below.
* `kernclust_template`: the template that contains the environmental variables for running kernel clustering commands. An example is provided in `/your/path/to/MedGP/scripts/kernclust_della.sh`. The generator will use this template to generate the commands to run the kernel clustering script for each cross-validation fold.
* `eval_template`: the template to run evaluations of the experiment. The generator will use this template to generate the commands to run evaluations (i.e computing patient-wise mean absolute error and coverage).
* `train_config`: a **list of dictinoaries** where each dictionary specifies one configuration of computing resources. Note that training is usually time consuming when the number of training points is large, and you might want to increase the number of threads if available. Here we provide an example dictionary (also avaialbe in `/your/path/to/MedGP/scripts/slurm_della.json`):
```
{
    "script_name": "train_medium.sh",
    "type": "slurm",
    "mem": "10000",
    "runtime": "2-00:00:00",
    "thread": 5,
    "host_name": "della",
    "host_thread_limit": 20,
    "min_mat_size": 500,
    "max_mat_size": 2000
}
```
In this case, we are telling the generator to create a bash executable script named `train_medium.sh` to work with slurm job scheduler, with memory limit as 10GB (10000MB) and runtime limit is 2 days. The number of threads is 5 and the maximum number of threads of the machine is 20. The script is applied the training cases where the total number of points is between 500 (`min_mat_size`) and 2000 (`max_mat_size`).
* `test_config`: the format is the same as `train_config`, but the setup here is applied to **testing**.

### (3) Put it all together
After preparing (1) and (2), you can generate your own experiment by running the generator. The detailed descriptions of arguments are provided in `medgpc.util.run_exp_generator.py`. Here, we provide an example script (also available in `/scripts/gen_medgpc_example.sh`) for how to pass important arguments at a glance:
```
python -m medgpc.util.run_exp_generator \
--path-config=path_della.json \
--hpc-config=slurm_della.json \
--feature-config=feature_all.json \
--cohort=heart_failure \
--cv-fold-num=10 \
--cv-seed=718 \
--exp-prefix=heart_failure_test \
--kernel=LMC-SM \
--prior=hier-gamma \
--Q=5 --R=8 --eta=0.01 --beta-lam=0.01 \
--kernel-cluster-alg=gmm \
--opt-config=opt_prior2.json \
--flag-plot-kernel-cluster=2
```
This command will generate an experiment of using our kernel described in the paper (`LMC-SM` kernel with hierarchical gamma prior `hier-gamma`) with 5 basis kernels and each is rank 8. The `\eta` parameter is 0.01 and `\beta_\lambda` parameter is 0.01.  We choose the algorithm for kernel clustering as Gaussian mixture model (`gmm`). Since we set `--exp-prefix=heart_failure_test`, the experiment will be named as `heart_failure_test_k7_q5_r8_p2_e0.01` and can be found under `exp_root_path` specified in `path_della.json`; more details of the naming rule can be found in `medgpc.util.run_exp_generator.py`. We will evaluate the performance using 10-fold cross-validation (`--cv-fold-num=10`) and the splits of fold is generated using random seed 718 (`--cv-seed=718`).

## Run your experiments
After you create your experiment, you can run the pipeline and collect the results with four scripts: 
* (i) `exp_root_path/(exp_name)/run_train_all.sh`: submit all training jobs to the cluster. This allows you to run training on each patient in parallel.
* (ii) `exp_root_path/(exp_name)/run_kernclust_all.sh`: submit all kernel clustering jobs to the cluster. This allows you to run kernel clustering on the level of cross-validation fold and extract one set of population-level kernels for each fold.
* (iii) `exp_root_path/(exp_name)/run_test_all.sh`: submit all testing jobs to the cluster. This allows you to run testing one-step-ahead imputation on each patient in parallel.
* (iv) `exp_root_path/(exp_name)/run_eval_all.sh`: submit evaluation jobs to the cluster; this allows you to collect sample-wise testing results into a vector of mean absolute error (MAE) and coverage for each feature. They will show up as `exp_root_path/(exp_name)/test/(test_mode)_feature(feature_index)_mae.bin` and `exp_root_path/(exp_name)/test/(test_mode)_feature(feature_index)_ci_ratio.bin`. `(test_mode)` is either `test_mean_wo_update` or `test_mean_w_update` for testing without or with momentum update respectively.
