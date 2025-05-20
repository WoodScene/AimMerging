# AIMMerging
Thank you for your interest in our work! This repository contains the original implementation of "AIMMerging: Leveraging Training Trajectories for Adaptive Iterative Model Merging in Language Model Continual Learning".

Reproducing the results from our paper is straightforward—just follow the steps outlined below.

## Local Setup
```
conda create -n AimMerging python=3.8
conda activate AimMerging
pip install -r requirements.txt
```

> **Important:**
> Please ensure the following package versions:
>
> * `transformers==4.28.1`
> * `peft==0.4.0`

Then, replace the corresponding files in the `transformers` package (typically located at `anaconda_path/envs/AimMerging/lib/python3.8/site-packages/transformers/`) with the modified versions of `trainer.py` and `training_args.py`.
These modifications are required to support our **Adaptive Iterative Model Merging (AIMMerging)** framework.

> Detailed comments are included in the modified files to help you understand the changes.

## Step 1. Preliminary Preparation
Our data preprocessing follows the approach used in [O-LoRA](https://github.com/cmnfriend/O-LoRA).
We also provide preprocessed datasets that are ready to use.

Download the required backbone models from Hugging Face:
* [T5-large](https://huggingface.co/google-t5/t5-large)
* [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
* [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
* [LLaMA2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)



## Step 2. Training and Inference
To fine-tune the models, run the following command. This will also generate predictions for three metrics: Overall Performance (OP), Forward Transfer (FWT), and Backward Transfer (BWT).
```ruby
./scripts/run_train_ours.sh
```

Notes:
* Use the `model_path` argument to specify the location of your downloaded models.
* We use [LoRA](https://github.com/microsoft/LoRA) for efficient parameter-efficient fine-tuning.
* Fine-tuned model weights will be saved to `$checkpoint_files`.
* **All the visualized results presented in our paper (Section "Visualization") will be saved in the `./Fig` folder for easy access.**
* The prediction results will be stored in the `$output` folder.


## Step 3. Evaluation
To calculate the metrics, run:
```ruby
./src/eval_avgPerf.py
./src/eval_fwt.py
./src/eval_bwt.py
```

We hope you find this repository useful! If you encounter any issues or have questions, feel free to open an issue or contact us.