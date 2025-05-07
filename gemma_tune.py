#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import itertools
import logging
import math
import os
import datetime
    
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from evaluate import load
import time
import string
import csv
import re
import random
import pandas as pd
import pickle
import json

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.utils.data import Dataset
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST
from retomaton import RetomatonWrapper

# from stats import datasetStats
import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    source_sufix: Optional[str] = field(
        default=None, metadata={"help": "A sufix to add before every source text (useful for T5 models)."}
    )
    eval_subset: str = field(default='validation')
    stride: int = field(default=512)
    patience: int = field(default=None)
    prompt: bool = field(default=False)
    post_process: bool = field(default=False)


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    dstore_damb: int = field(default=None, metadata={"help": "For the name of the dstore(avoiding ambuiguity)."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="checkpoints")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda1: float = field(default=0.25)
    lmbda2: float = field(default=0.1)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)
    t: int = field(default=10)

    ## RetoMaton args:
    retomaton: bool = field(default=False)
    cluster_dstore: bool = field(default=False)
    no_pointer: bool = field(default=False)
    min_knns: int = field(default=1)
    max_knns: int = field(default=1024)
    num_clusters: int = field(default=500000)
    sample_size: int = field(default=20000000)
    members: str = field(default=None)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")
    # logger.info(f"kNN parameters {knn_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    for split in list(raw_datasets.keys()):
        if split != data_args.eval_subset:
            del raw_datasets[split]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # gpu_id = 0
    # free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

    # print(f"Free memory: {free_mem / 1024 ** 2:.2f} MB")
    # print(f"Total memory: {total_mem / 1024 ** 2:.2f} MB")
    if model_args.model_name_or_path:
        # Use a pipeline as a high-level helper
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    if torch.cuda.is_available():
        model.to('cuda')
    logger.info(f'Model assigned to {model.device}')
    # print("Model assigned to ",model.module.device)
    model.resize_token_embeddings(len(tokenizer))
    # free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

    # print(f"Free memory: {free_mem / 1024 ** 2:.2f} MB")
    # print(f"Total memory: {total_mem / 1024 ** 2:.2f} MB")
    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    # print(model.forward.__code__.co_varnames)
    for split in list(raw_datasets.keys()):
        if split != data_args.eval_subset:
            del raw_datasets[split]


    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    def clean_text(text):
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove citations like [1], [citation needed]
        text = re.sub(r'\[[^\]]*\]', '', text)
    
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
    
        words = text.split()
        cleaned_words = []

        for word in words:
            # Remove word *if* it contains any non-ASCII characters
            if all(ord(char) < 128 for char in word):
                cleaned_words.append(word)
    
        # Strip leading/trailing whitespace
        return text.strip()
    

    def format_input(triviaq):
        quest = triviaq["question"]
        if quest.startswith("'") and quest.endswith("'"):
            quest = quest[1:-1]
        quest = re.sub(r'"+', "'", quest)
        prompt = '''Answer these questions:
Q: Who was President when the first Peanuts cartoon was published?
A: Harry Truman
Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
A: Sinclair Lewis
Q: Where in England was Dame Judi Dench born?
A: York
Q: William Christensen of Madison, New Jersey, has claimed to have the world's biggest collection of what?
A: Beer Cans
Q: In which decade did Billboard magazine first publish and American hit chart?
A: 30s
Q:'''
        triviaq['input_final_prompts'] = prompt + triviaq['question'] + '\nA:'
        return triviaq
    
    def process_gsm(answer):
        match = re.search(r"#### ([\d,]+\.?\d*)", answer)
        if match:
            value = match.group(1)
            return value.replace(',','')
        return answer

    def format_gsm(example):
        # Answer the following question through careful, concise step-by-step reasoning:\nQuestion: {question}\nSolution: {_target}
        # prompt = "<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOriginally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nShawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThere were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMichael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nOlivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
        # text = prompt + f"{example['question']}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = '''<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? <end_of_turn>
<start_of_turn>model
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
#### 6 <end_of_turn>
<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot? <end_of_turn>
<start_of_turn>model
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5 <end_of_turn>
<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? <end_of_turn>
<start_of_turn>model
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
#### 39 <end_of_turn>
<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny? <end_of_turn>
<start_of_turn>model
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
#### 8 <end_of_turn>
<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? <end_of_turn>
<start_of_turn>model
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
#### 9 <end_of_turn>
<start_of_turn> user
You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.
Problem: '''
        text = prompt + f"{example['question']} <end_of_turn>\n<start_of_turn>model\nAnswer: "
        example['inputs'] = text
        example['extracted_ans'] = process_gsm(example['answer'])
        return example

    def format_mmlu(mmlu):
        prefix = prompt_prefix[mmlu['subject']]
        mmlu['input'] = prefix +'\n\n' +mmlu['question'] + '\nA. ' + mmlu['choices'][0] + '\nB. ' + mmlu['choices'][1] + '\nC. ' + mmlu['choices'][2] + '\nD. ' + mmlu['choices'][3] #+ '\n'
        return mmlu
    
    def filter_by_word_count(example):
        return len(example['input'].split()) <= 1024

    if 'trivia' in data_args.dataset_name:
        with training_args.main_process_first(desc="format input prompt"):
            raw_datasets = raw_datasets.map(
                format_input,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="dataset map context",
            )
        raw_datasets = raw_datasets.remove_columns(['entity_pages','search_results'])
        column_names = raw_datasets[data_args.eval_subset].column_names
        text_column_name = 'input_final_prompts' if 'input_final_prompts' in column_names else column_names[0]
    elif 'gsm' in data_args.dataset_name:
        with training_args.main_process_first(desc="format input prompt"):
            raw_datasets = raw_datasets.map(
                format_gsm,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="format input",
            )
        column_names = raw_datasets[data_args.eval_subset].column_names
        print(column_names)
        text_column_name = 'inputs' if 'inputs' in column_names else column_names[0]
    elif 'mmlu' in data_args.dataset_name:
        with open("mmlu_prompts.json", "r") as f:
            prompt_prefix = json.load(f)
        with training_args.main_process_first(desc="format input prompt"):
            raw_datasets = raw_datasets.map(
                format_mmlu,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="format input",
            )
        raw_datasets = raw_datasets.filter(filter_by_word_count)
        column_names = raw_datasets[data_args.eval_subset].column_names
        print(column_names)
        text_column_name = 'input' if 'input' in column_names else column_names[0]

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if knn_args.retomaton or knn_args.cluster_dstore:
        knn_wrapper = RetomatonWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, t=knn_args.t, lmbda1=knn_args.lmbda1, lmbda2=knn_args.lmbda2, knn_temp=knn_args.knn_temp, probe=knn_args.probe,
            no_pointer=knn_args.no_pointer, min_knns=knn_args.min_knns, max_knns=knn_args.max_knns,
            members=knn_args.members)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, t=knn_args.t, lmbda1=knn_args.lmbda1, lmbda2=knn_args.lmbda2, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)
        if knn_args.dstore_size is not None:
            knn_wrapper.dstore_size, knn_wrapper.dstore_damb = knn_args.dstore_size, None
            knn_wrapper.index, knn_wrapper.keys, knn_wrapper.vals = knn_wrapper.setup_faiss()
            knn_wrapper.unique = True

    def postprocess_trivia(answer):
        if answer == '': 
            return answer
        # words = answer.split()
        # filtered_words = [word for word in words if re.fullmatch(r'[a-zA-Z0-9.,!?\'\"-]+', word)]
        # answer = ' '.join(filtered_words)
        answer = ' '.join(answer.split('.')[:-1])
        answer = answer.strip()
        answer = answer.lower().replace('-',' ') #splits hyphenated text
        repl_table = string.punctuation.maketrans("", "", string.punctuation) #remove puntuation marks
        answer = answer.translate(repl_table)
        words = answer.split() 
        filtered_words = [word for word in words if word not in ['a', 'the', 'an']] 
        answer =  ' '.join(filtered_words)
        return answer

    def process_mmlu(answer):
        match = re.search(r"Answer:\s*([A-Z])", answer)
        if match:
            value = match.group(1)
            if value == 'A':
                return 0
            if value == 'B':
                return 1
            if value == 'C':
                return 2
            if value == 'D':
                return 3
        return answer

    def match(pred, answers):
        if pred in answers:
            return 1
        return 0
    
    def f1_score(pred, answer):
        f1 = 0
        pred_unigrams = set(pred.split(' '))     
        for ans in answer:
            ans_unigrams = set(ans.split(' '))
            tp = len(pred_unigrams & ans_unigrams)
            fp = len(pred_unigrams - ans_unigrams)
            fn = len(ans_unigrams - pred_unigrams)
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            if tp == 0:
                continue
            else:
                cur_f1 = 2*(precision*recall)/(precision+recall)  
            if cur_f1 > f1:
                f1 = cur_f1
        return f1

    param_df = pd.read_csv('params.csv', header=0)
    param_df = param_df.fillna(0)

    for k in param_df['k']:
        if k == 0:
            continue
        for lmbda1 in param_df['lmbda1'][1:]:
            if lmbda1 == 0:
                continue
            for knn_temp in param_df['knn_temp']:
                if knn_temp == 0:
                    continue
                knn_wrapper.lmbda1 = lmbda1
                knn_args.lmbda1 = lmbda1
                knn_wrapper.knn_temp = knn_temp
                knn_args.knn_temp = knn_temp
                knn_wrapper.k = int(k)
                knn_args.k = int(k)
                if data_args.prompt:
                    logger.info(f'lambda1: {knn_wrapper.lmbda1} knn_temp: {knn_wrapper.knn_temp} k: {knn_wrapper.k}')
                    batch_size = 2 if knn_wrapper is not None else 16
                    outputs = []
                    num_beams = 5
                    if 'trivia' in data_args.dataset_name:
                        max_new_tokens = 24 
                    elif 'gsm' in  data_args.dataset_name:
                        max_new_tokens = 512
                    elif 'mmlu' in  data_args.dataset_name:
                        max_new_tokens = 10
                    elif 'eval' in  data_args.dataset_name:
                        max_new_tokens = 10
                    if knn_wrapper is not None:
                        knn_wrapper.batch_size = batch_size
                        knn_wrapper.max_new_tokens = max_new_tokens
                        logger.info(f'max_new_tokens: {knn_wrapper.max_new_tokens}')
                        knn_wrapper.num_beams = num_beams
                        if 'trivia' in data_args.dataset_name:
                            dstore_path = knn_args.dstore_dir + '/0rc.pkl'
                            with open(dstore_path, 'rb') as file:
                                knn_wrapper.dstore_sizes = pickle.load(file)
                    
                    tokenizer.padding_side='left'
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    
                    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
                    
                    # raw_datasets[data_args.eval_subset] = raw_datasets[data_args.eval_subset].select(range(144,1000))
                    logger.info(f'[{data_args.eval_subset}] has {raw_datasets[data_args.eval_subset].num_rows}')
                    input_dataset = raw_datasets[data_args.eval_subset][text_column_name]
                    input_dataset = ListDataset(input_dataset)
                    start = datetime.datetime.now()
                    
                    logger.info(f'Running generation pipeline')
                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    #     with record_function("model_inference"):
                            # model(inputs)
                    outputs = []
                    for out in tqdm(text_generator(input_dataset, batch_size = batch_size, max_new_tokens=max_new_tokens, pad_token_id = tokenizer.eos_token_id, num_beams = num_beams), total = len(input_dataset)):
                        # print(len(out))
                        outputs = outputs + out
                        torch.cuda.empty_cache()
                    logger.info(f'Generating answers took {datetime.datetime.now() - start} s')
                    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    # print(outputs)
                    outputs = [op['generated_text'] for op in outputs]
                    
                    answers = []
                    for op, row in zip(outputs, raw_datasets[data_args.eval_subset][text_column_name]):
                        answer = op[len(row):]
                        # answer = answer.split('\n')[0]
                        answers.append(answer)
                    if knn_args.dstore_size is None:
                        knn_args.dstore_size = 0
                        knn_args.lmbda2 = 0
                        knn_args.t = 0
                    outputs = answers
                    answers = []
                    if 'trivia' in data_args.dataset_name:
                        answers = [postprocess_trivia(op.split('\n')[0]) for op in outputs]
                        # Calculating metrics
                        match_list = [match(pred, ans['normalized_aliases']) for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['answer'])]
                        metric_em = sum(match_list)/raw_datasets[data_args.eval_subset].num_rows*100
                        # print(f'{sum(match_list)}/{raw_datasets[data_args.eval_subset].num_rows},{metric_em}')
                        f1_list = [f1_score(pred, ans['normalized_aliases']) for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['answer'])]
                        metric_f1 = sum(f1_list)/raw_datasets[data_args.eval_subset].num_rows*100
                        if not os.path.exists(training_args.output_dir):
                            os.makedirs(training_args.output_dir)
                        time = datetime.datetime.now()
                        path = training_args.output_dir+"/output"+ time.strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
                        with open(path, mode='w', newline='', encoding='utf-8') as file1:
                            writer = csv.DictWriter(file1, fieldnames=['input', 'output', 'question', 'correct_answers', 'answer', 'exact_match', 'f1'])
                            writer.writeheader()
                            for row, outp, answ, em, f1  in zip(raw_datasets[data_args.eval_subset], outputs, answers, match_list, f1_list):
                                writer.writerow({'input': row[text_column_name], 'output': outp, 'question' : row['question'], 'correct_answers' : row['answer']['normalized_aliases'], 'answer': answ, 'exact_match': em, 'f1': f1})
                        
                        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
                        
                        fieldnames = ['time', 'model', 'dataset', 'retomaton', 'knn', 'lambda1', 'lambda2', 'block_size', 'k','knn_temp', 't', 'max_new_tokens', 'dstore_size', 'exact_match','f1', 'notes']
                        metrics_to_file = { 'time': time_st,
                            'model': model_args.model_name_or_path,
                            'dataset': data_args.dataset_name,
                            'retomaton': knn_args.retomaton, 
                            'knn': knn_args.knn, 
                            'lambda1': knn_args.lmbda1,
                            'lambda2' : knn_args.lmbda2, 
                            'block_size': block_size, 
                            'k': knn_args.k,
                            'knn_temp': knn_args.knn_temp,
                            't' : knn_args.t,
                            'max_new_tokens' : max_new_tokens,
                            'dstore_size' : knn_args.dstore_size,
                            'exact_match' : round(metric_em, 2),
                            'f1' : round(metric_f1,2),
                            'notes':  'sample',
                        }
                    elif 'gsm' in data_args.dataset_name:
                        answers = [process_gsm(op) for op in outputs]
                        # for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['extracted_ans']):
                        #     print(pred, ans, pred==ans)
                        acc_list = [1 if ans==pred else 0 for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['extracted_ans'])]
                        accuracy = sum(acc_list)/raw_datasets[data_args.eval_subset].num_rows*100
                        if not os.path.exists(training_args.output_dir):
                            os.makedirs(training_args.output_dir)
                        time = datetime.datetime.now()
                        path = training_args.output_dir+"/output"+ time.strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
                        with open(path, mode='w', newline='', encoding='utf-8') as file1:
                            writer = csv.DictWriter(file1, fieldnames=['input', 'output', 'question', 'answer', 'pred', 'accuracy'])
                            writer.writeheader()
                            for row, outp, answ, acc  in zip(raw_datasets[data_args.eval_subset], outputs, answers, acc_list):
                                writer.writerow({'input': row['inputs'], 'output': outp, 'question' : row['question'], 'answer' : row['answer'], 'pred': answ, 'accuracy': acc})
                        
                        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
                        
                        fieldnames = ['time', 'model', 'dataset', 'dstore_path', 'knn', 'retomaton', 'lambda', 'k','knn_temp', 'dstore_size', 'accuracy', 'notes']
                        metrics_to_file = metrics_to_file = { 'time': time_st,
                            'model': model_args.model_name_or_path,
                            'dataset': data_args.dataset_name,
                            'dstore_path': knn_args.dstore_dir if knn_wrapper is not None else 'None',
                            'knn': knn_args.knn, 
                            'retomaton': knn_args.retomaton,
                            'lambda': knn_args.lmbda1 if knn_wrapper is not None else 0, 
                            'k': knn_args.k if knn_wrapper is not None else 0,
                            'knn_temp': knn_args.knn_temp if knn_wrapper is not None else 0,
                            'dstore_size' : knn_args.dstore_size if knn_wrapper is not None else 0,
                            'accuracy' : round(accuracy, 2),
                            'notes':  'kNN_' + str(len(input_dataset)) if knn_wrapper is not None else 'Base_' + str(len(input_dataset)),
                        }
                    elif 'mmlu' in data_args.dataset_name:
                        answers = [process_mmlu(op) for op in outputs]
                        acc_list = [1 if ans==pred else 0 for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['answer'])]
                        accuracy = sum(acc_list)/raw_datasets[data_args.eval_subset].num_rows*100
                        if not os.path.exists(training_args.output_dir):
                            os.makedirs(training_args.output_dir)
                        time = datetime.datetime.now()
                        path = training_args.output_dir+"/output"+ time.strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
                        with open(path, mode='w', newline='', encoding='utf-8') as file1:
                            writer = csv.DictWriter(file1, fieldnames=['input', 'output', 'question', 'answer', 'pred', 'accuracy'])
                            writer.writeheader()
                            for row, outp, answ, acc  in zip(raw_datasets[data_args.eval_subset], outputs, answers, acc_list):
                                writer.writerow({'input': row[text_column_name], 'output': outp, 'question' : row['question'], 'answer' : row['answer'], 'pred': answ, 'accuracy': acc})
                        
                        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
                        
                        fieldnames = ['time', 'model', 'dataset', 'dstore_path' ,'knn', 'retomaton','lambda', 'k','knn_temp', 'dstore_size', 'accuracy', 'notes']
                        metrics_to_file = { 'time': time_st,
                            'model': model_args.model_name_or_path,
                            'dataset': data_args.dataset_name,
                            'dstore_path': knn_args.dstore_dir if knn_wrapper is not None else 'None',
                            'knn': knn_args.knn, 
                            'retomaton': knn_args.retomaton,
                            'lambda': knn_args.lmbda1 if knn_wrapper is not None else 0, 
                            'k': knn_args.k if knn_wrapper is not None else 0,
                            'knn_temp': knn_args.knn_temp if knn_wrapper is not None else 0,
                            'dstore_size' : knn_args.dstore_size if knn_wrapper is not None else 0,
                            'accuracy' : round(accuracy, 2),
                            'notes':  'kNN/reto_' + str(len(input_dataset)) if knn_wrapper is not None else 'Base_' + str(len(input_dataset)),
                        }
                    logger.info(metrics_to_file)
                    fieldnames = list(metrics_to_file.keys())
                    csv_file =  training_args.output_dir+"/runs.csv"


                    if not os.path.isfile(csv_file):
                        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                            writer = csv.DictWriter(file, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerow(metrics_to_file)
                    else:
                        with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                            writer = csv.DictWriter(file, fieldnames=fieldnames)
                            writer.writerow(metrics_to_file)

    if knn_args.build_index:
        knn_wrapper.build_index()

    if knn_args.cluster_dstore:
        knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()
    
if __name__ == "__main__":
    main()

