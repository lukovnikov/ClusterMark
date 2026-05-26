# Towards Robust Red-Green Watermarks for Autoregressive Image Generators
## Project structure
The code consists of three parts: 
1. a project containing shared code, used for verification (`wartermarks`)
2. a project modified from the `1d-tokenizer` codebase for the RAR experiments
3. a project modified from the `LlamaGen` codebase for LlamaGen experiments

## Installation

### Step 1: clone this repo with the submodules:
'>> git clone --recurse-submodules github.com/lukovnikov/ClusterMark'
### Installing RAR
* Clone or copy the `1d-tokenizer` code into a directory and change into it
* create a new environment: `>> python -m venv venv` inside the project
* Follow the installation instructions in the original `1d-tokenizer` repo to setup the environment
### Installing LlamaGen
* Clone or copy the `LlamaGen` code into a directory and change into it
* create a new environment: `>> python -m venv venv` inside the project
* Follow the installation instructions in the original `LlamaGen` repo to setup the environment

### Installing shared project (wartermarks)
* Clone or copy the `wartermarks` project and change into it
* Install the shared project in both RAR and LlamaGen projects:
	*  activate base project environment: `>> source ../1d-tokenizer/venv/bin/activate` for RAR)
	* Install the shared project code in development mode: `>> pip install -e .`

## Generating (watermarked) images
### RAR:
In the `1d-tokenizer` project, two Python scripts are provided:
* `sample_c2i.py` for unwatermarked generation
* `sample_c2i_wm.py` for watermarked generation

To generate 2000 **watermarked** images **with 64 clusters**, watermark strength $\delta=5$, prefix $\kappa = 0$ and green token fraction $\gamma = 0.25$, run the following command:
`python sample_c2i_wm.py --num_samples 2000 --num_clusters 64 --wm_seed_prefix 0 --wm_red_penalty 5 --wm_green_fraction 0.25`

To generate **watermarked** images **without clusters**:
`python sample_c2i_wm.py --num_samples 2000 --num_clusters 0 --wm_seed_prefix 0 --wm_red_penalty 5 --wm_green_fraction 0.25`

Default batch sizes are tuned for a GPU with 48GB memory but can be adjusted with `--batsize <BATCH SIZE>`.

### LlamaGen:
TODO

## Evaluation of watermark detection
Both `1d-tokenizer` and `LlamaGen` projects contain a file `evaluate.py`, which implements robustness evaluation, FID computation (using `cleanfid`) and computing finally reported metrics.

To compute *theoretical* TPR's, run the following command:
`python evaluate.py one --dir <EXPDIR> --perturbationset <fullchallenge|regen>`

Here, `<EXPDIR>` would look (after following the generation instructions above) like `experiments_v1/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.5_penalty5_prefix0/` .

To compute empirical TPR's and AUROC's as reported in the paper, follow these steps:
1. evaluate watermarked data with `evaluate.py one --dir <EXPDIR>` 
2. evaluate *unwatermarked generated*  data with `evaluate.py one --perturbationset negative --dir <EXPDIR> --imgdir <CLEANDIR>` -- this evaluates a set of images stored in `<CLEANDIR>` with the watermarking settings from `<EXPDIR>` and stores the outcomes in `<EXPDIR>`.
3. compute empirical TPR and AUROC with `evaluate.py roc_one --dir <EXPDIR> --posfile <POSFILE> --negfile <NEGFILE>` . For a perturbationset `fullchallenge`, posfile should be named like `results_v3_evalfirst2000_perturb=fullchallenge.json` and negfile should be named like `results_v3_evalfirst2000_perturb=negative.json` . The results are stored in `roc_summary_v3_evalfirst2000_perturb\=fullchallenge.json` .

To compute FID (this requires 50k samples), run the following command:
`python evaluate.py cleanfid_one --generated_path <EXPDIR> --real_path <IMAGENET_VAL_DIR>`

## Cluster predictor training
The code to train the cluster predictor can be found in `clusterpred.py`.
Pretrained models will be provided (TODO)

## Cluster predictor use
Once the cluster predictor is trained, it can be used by specifying `--clusterpredictor <CLUSTERPREDCKPT> --useclusterpredictor` in the evaluation script. For example:
`python evaluate.py one --dir <EXPDIR> --perturbationset fullchallenge --clusterpredictor <CLUSTERPREDCKPT> --useclusterpredictor` .



