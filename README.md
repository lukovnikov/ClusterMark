
# Towards Robust Red-Green Watermarks for Autoregressive Image Generators

## Installation

(reproduced with Python 3.13)

### Step 1: clone this repo with the submodules:

`git clone --recurse-submodules https://github.com/lukovnikov/ClusterMark`

### Step 2: setup the environment for the chosen generator:
For `1d-tokenizer`:
1. `cd clustermark_1d-tokenizer`
2. `python -m venv .venv`
3. `source activate .venv/bin/python`

Follow installation instructions in `1d-tokenizer` to setup the environment for this generator:

`pip install -r requirements.txt`

The RAR-XL checkpoint as well as the VAE (maskgit-vqgan-imagenet-f16-256.bin) are downloaded automatically when you first run the generation script and placed in `ckpts`.

### Step 3: setup `wartermarks` in the generator's environment:
While in the generator's environment, install the shared project (wartermarks):

`cd ../clustermark_wartermarks`

`pip install -e .`

## Generating (watermarked) images

### RAR:

In the `1d-tokenizer` project, two Python scripts are provided:

*  `sample_c2i.py` for unwatermarked generation

*  `sample_c2i_wm.py` for watermarked generation

  

To generate 2000 **watermarked** images **with 64 clusters**, watermark strength $\delta=5$, prefix $\kappa = 0$ and green token fraction $\gamma = 0.25$, run the following command:

`python sample_c2i_wm.py --num_samples 2000 --num_clusters 64 --wm_seed_prefix 0 --wm_red_penalty 5 --wm_green_fraction 0.25`

  

To generate **watermarked** images **without clusters**:

`python sample_c2i_wm.py --num_samples 2000 --num_clusters 0 --wm_seed_prefix 0 --wm_red_penalty 5 --wm_green_fraction 0.25`


  

### LlamaGen:

TODO

  

## Evaluation of watermark detection

Both `1d-tokenizer` and `LlamaGen` projects contain a file `evaluate.py`, which implements robustness evaluation, FID computation (using `cleanfid`) and computing finally reported metrics.

  

To compute *theoretical* TPR's, run the following command:

`python evaluate.py one --dir <EXPDIR> --perturbationset <fullchallenge|regen>`

Here, `<EXPDIR>` would look (after following the generation instructions above) like `experiments_v1/gen_wm_v1_2000samples_rar_xl_64clusters_greenfrac0.5_penalty5_prefix0/` .


To use the token- or cluster-predictor, add `--usetokenpredictor` (`--useclusterpredictor`) and make sure `--tokenpredictor` (`--clusterpredictor`) points to the correct location. 
For example:
`python evaluate.py one --dir <EXPDIR> --perturbationset fullchallenge --clusterpredictor <CLUSTERPREDCKPT> --useclusterpredictor` .
You can download checkpoints here: [denisl/clustermark_1d_tokenizer](https://huggingface.co/denisl/clustermark_1d_tokenizer/tree/main)

  

To compute empirical TPR's and AUROC's as reported in the paper, follow these steps:

1. evaluate watermarked data with `evaluate.py one --dir <EXPDIR>`

2. evaluate *unwatermarked generated* data with `evaluate.py one --perturbationset negative --dir <EXPDIR> --imgdir <CLEANDIR>` -- this evaluates a set of images stored in `<CLEANDIR>` with the watermarking settings from `<EXPDIR>` and stores the outcomes in `<EXPDIR>`.

3. compute empirical TPR and AUROC with `evaluate.py roc_one --dir <EXPDIR> --posfile <POSFILE> --negfile <NEGFILE>` . For a perturbationset `fullchallenge`, posfile should be named like `results_v3_evalfirst2000_perturb=fullchallenge.json` and negfile should be named like `results_v3_evalfirst2000_perturb=negative.json` . The results are stored in `roc_summary_v3_evalfirst2000_perturb\=fullchallenge.json` .

  

To compute FID (this requires 50k samples), run the following command:

`python evaluate.py cleanfid_one --generated_path <EXPDIR> --real_path <IMAGENET_VAL_DIR>`

  

## Cluster predictor training

The code to train the cluster predictor (or token predictor) can be found in `clusterpred.py` (`tokenpred.py`) in the `wartermarks` repo.

Pretrained models are provided at [denisl/clustermark_1d_tokenizer](https://huggingface.co/denisl/clustermark_1d_tokenizer/tree/main).

