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
import copy
import os
import datetime
    
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from evaluate import load
import time
import string
import re
import csv

import datasets
from datasets import load_dataset, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import pickle
import pandas as pd

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
    DataCollatorWithPadding,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST
from retomaton_old import RetomatonWrapper

# from stats import datasetStats
import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

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
    # do_predict: bool = field(default=False)
    prompt: bool = field(default=False)


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
        # if "validation" not in raw_datasets.keys() and 'gsm' not in data_args.dataset_name:
        #     raw_datasets["validation"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         cache_dir=model_args.cache_dir,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         cache_dir=model_args.cache_dir,
        #     )
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
        # If not training and not evaluating on train, we do not need to process it
        if split != data_args.eval_subset:
            del raw_datasets[split]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    gpu_id = 0
    free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

    print(f"Free memory: {free_mem / 1024 ** 2:.2f} MB")
    print(f"Total memory: {total_mem / 1024 ** 2:.2f} MB")
    
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
    tokenizer.pad_token = tokenizer.eos_token 
    if model_args.model_name_or_path:
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

    model.to('cuda')
    print("Model assigned to ",model.device)
    model.resize_token_embeddings(len(tokenizer))

    # free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

    # print(f"Free memory: {free_mem / 1024 ** 2:.2f} MB")
    # print(f"Total memory: {total_mem / 1024 ** 2:.2f} MB")

    # Injecting KNN
    dimension = model.config.hidden_size
    logger.info(f'Dimension: {dimension}')
    knn_wrapper = None
    knn_args.seed = training_args.seed

    # print(model.forward.__code__.co_varnames)


    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
    def clean_text(text):
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove citations like [1], [citation needed]
        text = re.sub(r'\[[^\]]*\]', '', text)
    
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
    
        # Optionally remove non-ASCII characters (can disable this if you want accents preserved)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
    
        # Strip leading/trailing whitespace
        return text.strip()
    
    def clean_example(example):
        example['question'] = clean_text(example['question'])
        example['context'] = clean_text(' '.join(example['entity_pages']['wiki_context'] + example['search_results']['search_context']))
    
        return example

    def format_QA(example):
        # Answer the following question through careful, concise step-by-step reasoning:\nQuestion: {question}\nSolution: {_target}
        # text = f"Answer the following question through careful, concise step-by-step reasoning:\nQuestion: {example['question']}\nSolution: {example['answer']}"
        if 'gemma' in model_args.model_name_or_path:
            print('gemma')
            text = f"You are a helpful 2nd-grade math teacher. Help a 2nd grader to answer problem in a short and clear manner. Your response should end with \"#### [NUM]\" where [num] is the response to the problem.\nProblem: {example['question']}\nAnswer: {example['question']}"    
        else:
            text = f"<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {example['question']}\nYour response should end with \"\n#### [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{example['answer']}\n"
        # text = re.sub(r'\s+', ' ', text)
        example['inputs'] = text
        # print(example)
        return example
    
    def format_gsm(example):
        # prompt = "Given the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nOriginally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nShawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nThere were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nMichael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\nOlivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
        # text = prompt + f"{example['question']}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n\n"
        prompt = "<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nThere are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nThere are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nOriginally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nJason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nShawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nThere were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nMichael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\nOlivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
        text = prompt + f"{example['question']}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{example['answer']}"
        example['inputs'] = text
        # match = re.search(r"#### (\d+)", example['answer'])
        # if match:
        #     example['extracted_ans'] = match.group(1)
        # else:
        #     example['extracted_ans'] = ''
        return example

    def filter_by_word_count(example):
        return len(example['question'].split()) <= 2048

    def format_mmlu(mmlu):
        option = {0:'A',
            1:'B',
            2:'C',
            3:'D'
        }
        mmlu['option'] = option[mmlu['answer']]
        mmlu['inputs'] = f"Answer the following multiple choice question. Choose the correct answer by selecting the letter only (A, B, C, or D).\n{mmlu['question']}\nA. {mmlu['choices'][0]}\nB. {mmlu['choices'][1]}\nC. {mmlu['choices'][2]}\nD. {mmlu['choices'][3]}\nAnswer: {mmlu['option']}"
        # print(mmlu)
        # print(1/0)
        return mmlu

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = examples if isinstance(examples['input_ids'][0], int) else {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        input_ids = []
        attention_mask = []
        labels = []
        # We implement a sliding window, so all tokens have a non-zero context in their prediction.
        # We then mask the duplicate tokens' labels, to not count any token twice in the loss.
        for i in tqdm(range(0, total_length, data_args.stride),total = total_length):
            begin_loc = max(i + data_args.stride - block_size, 0)
            end_loc = min(i + data_args.stride, total_length)
            trg_len = end_loc - i
            cur_input_ids = concatenated_examples['input_ids'][begin_loc:end_loc]
            cur_labels = list(cur_input_ids)
            cur_labels[:-trg_len] = [padding_index] * (len(cur_labels) - trg_len)

            if len(cur_input_ids) < block_size:
                padding_size = block_size - len(cur_input_ids)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                cur_input_ids += [pad_token_id] * padding_size
                cur_labels += [padding_index] * padding_size
            
            input_ids.append(cur_input_ids)
            attention_mask.append([1] * len(cur_labels))
            labels.append(cur_labels)

        result = {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        return result

    def pad_labels(example):
        # input_id_len = len(example['input_ids'])
        # print(f"Before: {example['input_ids']}")
        padding_size = max_len_pad - len(example['input_ids'])
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_input_seq = [pad_token_id] * padding_size
        pad_label_seq = [padding_index] * padding_size
        pad_att_seq = [0] * padding_size
        example['labels'] = pad_label_seq + example['input_ids']
        example['input_ids'] = pad_input_seq + example['input_ids']
        example['attention_mask'] = pad_att_seq + example['attention_mask']
        # print(f"After: {example['input_ids']}")
        return example

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

    if 'triviaqa' in data_args.dataset_name and training_args.do_eval:
        with training_args.main_process_first(desc="clean context"):
            raw_datasets = raw_datasets.map(
                clean_example,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="dataset map context",
            )
        column_names = raw_datasets[data_args.eval_subset].column_names
        text_column_name = "context" if "context" in column_names else column_names[0]
        batched = False
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif 'gsm' in data_args.dataset_name and training_args.do_eval:
        # raw_datasets['test'] = raw_datasets['test'].select(range(16))
        with training_args.main_process_first(desc="format_gsm"):
            raw_datasets = raw_datasets.map(
                format_gsm,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="dataset map context",
            )
        logger.info(f'[{data_args.eval_subset}] has {raw_datasets[data_args.eval_subset].num_rows}')
        column_names = raw_datasets[data_args.eval_subset].column_names
        text_column_name = "inputs" if "inputs" in column_names else column_names[0]
        tokenizer.padding_side = "left"
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        max_len_pad = max(len(input_ids) for input_ids in tokenized_datasets[data_args.eval_subset]['input_ids'])
        logger.info(f'[{split}] Max length of input: {max_len_pad}')
    
    elif 'mmlu' in data_args.dataset_name and training_args.do_eval:
        logger.info(f'[{data_args.eval_subset}] has {raw_datasets[data_args.eval_subset].num_rows}')
        raw_datasets = raw_datasets.filter(filter_by_word_count)
        with training_args.main_process_first(desc="clean context"):
            raw_datasets = raw_datasets.map(
                format_mmlu,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="dataset map context",
            )
        logger.info(f'[{data_args.eval_subset}] has {raw_datasets[data_args.eval_subset].num_rows}')
        column_names = raw_datasets[data_args.eval_subset].column_names
        text_column_name = "inputs" if "inputs" in column_names else column_names[0]
        logger.info(f'Formatted input: {raw_datasets[data_args.eval_subset][text_column_name][0]}')
        # print(1/0)
        tokenizer.padding_side = "left"
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        max_len_pad = max(len(input_ids) for input_ids in tokenized_datasets[data_args.eval_subset]['input_ids'])
        logger.info(f'[{split}] Max length of input: {max_len_pad}')

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    if 'triviaqa' in data_args.dataset_name and training_args.do_eval:
        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
    elif ('gsm' in data_args.dataset_name or 'mmlu' in data_args.dataset_name) and training_args.do_eval:
        with training_args.main_process_first(desc="generating labels"):
            lm_datasets = tokenized_datasets.map(
                pad_labels,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"labels",
            )
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    elif training_args.do_eval:
        if data_args.eval_subset not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        print("eval dataset picked with cols", eval_dataset.column_names)

    if knn_args.retomaton or knn_args.cluster_dstore:
        knn_wrapper = RetomatonWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda1=knn_args.lmbda1, knn_temp=knn_args.knn_temp, probe=knn_args.probe,
            no_pointer=knn_args.no_pointer, min_knns=knn_args.min_knns, max_knns=knn_args.max_knns,
            members=knn_args.members)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda1=knn_args.lmbda1, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    if knn_args.cluster_dstore:
            knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)
        if knn_args.dstore_size is not None:
            knn_wrapper.dstore_size, knn_wrapper.dstore_damb = knn_args.dstore_size, None
            knn_wrapper.reconstruct_index, knn_wrapper.index, knn_wrapper.keys, knn_wrapper.vals = knn_wrapper.setup_faiss()
            knn_wrapper.unique = True
        if knn_args.retomaton:
            knn_wrapper.load_retomaton()

    def do_train():
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def do_eval():
        logger.info("*** Evaluate ***")
        training_args.per_device_eval_batch_size=1
        metrics = trainer.evaluate()
        
        logger.info('Evaluation complete, calculating perplexity')

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(trainer.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        if knn_wrapper is not None:
            knn_metrics = knn_wrapper.get_metrics()
            metrics.update(knn_metrics)
            metrics.update({'kld': knn_wrapper.kl_reto})

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
        write_metrics = { 'time': time_st,
            'model' : model_args.model_name_or_path,
            'dataset' : data_args.dataset_name,
            'dstore_path' : knn_args.dstore_dir,
            'dstore_size' : knn_args.dstore_size,
            'lambda' : knn_wrapper.lmbda1,
            'k' : knn_wrapper.k,
            'knn_temp' : knn_wrapper.knn_temperature,
            'ppl' : perplexity,
            'kld' : knn_wrapper.kl_reto/total_eval_tokens,
            'nll' : knn_wrapper.nll/total_eval_tokens,
            'notes' : 'kld_nll_' + str(raw_datasets[data_args.eval_subset].num_rows),
        }
        logger.info(write_metrics)
        fieldnames = list(write_metrics.keys())
        csv_file = training_args.output_dir+"/kld.csv"

        if not os.path.isfile(csv_file):
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_metrics)
        else:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(write_metrics)

    
    def do_predict():
        logger.info("*** Predict ***")

        output = trainer.predict(eval_dataset)
        token_ids = np.argmax(output.predictions, axis=-1)
        # print(token_ids)
        decoded_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        print(decoded_texts)
    if training_args.do_eval or training_args.do_train:
        if 'trivia' in data_args.dataset_name:
            dstore_path = knn_args.dstore_dir + '/0rc.pkl'
            if os.path.exists(dstore_path):
                with open(dstore_path, 'rb') as file:
                    dstore_sizes = pickle.load(file)
            else:
                dstore_sizes = []
            print(f'# of dstores: {len(dstore_sizes)}')

            logger.info(f'Fetching duplicates')
            df = pd.DataFrame(raw_datasets[data_args.eval_subset])
            context = df[text_column_name]
            first_occurrence = {}
            dups = [0] * len(df)

            for i, text in enumerate(context):
                if isinstance(text, list):
                    text = ' '.join(text)
                if text in first_occurrence:
                    dups[i] = first_occurrence[text]
                else:
                    first_occurrence[text] = i
            logger.info(f'setting up {len(first_occurrence)} dstores')
            del raw_datasets, tokenized_datasets, 
            
            for rowind in tqdm(range(len(dstore_sizes), eval_dataset.num_rows), desc='Local Dstore'):
                logger.info(f'{len(dstore_sizes), rowind}')
                if dups[rowind]!=0:
                    dstore_sizes.append(dstore_sizes[dups[rowind]])
                    logger.info(f'Duplicate found - {dstore_sizes[dups[rowind]]}! Skipping')
                    with open(dstore_path, 'wb') as file:
                        pickle.dump(dstore_sizes, file)
                    continue

                knn_args.dstore_size = None
                knn_args.dstore_damb = None

                eval_input = eval_dataset[rowind]
                eval_input = Dataset.from_dict({k: v for k, v in eval_input.items()})
                # Initialize our Trainer
                training_args.remove_unused_columns=False
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset if training_args.do_train else None,
                    eval_dataset=eval_input if training_args.do_eval else None,
                    tokenizer=tokenizer,
                    # Data collator will default to DataCollatorWithPadding, so we change it.
                    data_collator=default_data_collator,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
                )
                # print(eval_input['labels'])
                trainer.args._n_gpu = 1
                total_eval_tokens = 0
                for chunk in eval_input['labels']:
                    # print(f"raw chunk: {chunk}")
                    total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
                    # print(f"tokens: {[x for x in chunk[1:] if x != padding_index]}")
                logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')
                # print(1/0)
                # if knn_args.dstore_size is None and split == 'train':
                knn_args.dstore_size = total_eval_tokens
                print('Dstore size: ', knn_args.dstore_size)

                while (knn_args.dstore_size, knn_args.dstore_damb) in dstore_sizes:
                    if knn_args.dstore_damb is None:
                        knn_args.dstore_damb = 1
                        continue
                    knn_args.dstore_damb+=1
                dstore_sizes.append((knn_args.dstore_size,knn_args.dstore_damb))
                # dstore_sizes[rowind] = (knn_args.dstore_size,knn_args.dstore_damb)
                
                if knn_wrapper is not None:
                    knn_wrapper.dstore_size, knn_wrapper.dstore_damb = knn_args.dstore_size,knn_args.dstore_damb
                    knn_wrapper.break_into(model)
                
                # Training
                if training_args.do_train:
                    do_train()

                # Evaluation
                if training_args.do_eval:
                    do_eval()

                if knn_args.build_index:
                    knn_wrapper.build_index()

                if knn_args.cluster_dstore:
                    knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
                
                if knn_wrapper is not None:
                    knn_wrapper.break_out()
                with open(dstore_path, 'wb') as file:
                    pickle.dump(dstore_sizes, file)
        
        elif 'gsm' in data_args.dataset_name or 'mmlu' in data_args.dataset_name:
            tokenizer.padding_side = "left"
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
            )
            trainer.args._n_gpu = 1
            
            for split, data in lm_datasets.items():
                total_eval_tokens = 0    
                max_len = 0    
                for chunk in data['labels']:
                    total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
                    # max_len = len(chunk) if len(chunk)>max_len else max_len
                logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')
                max_len = max(len(input_ids) for input_ids in tokenized_datasets[split]['input_ids'])
            logger.info(f'[{split}] Max length of input: {max_len}')
            if knn_args.dstore_size is None and split == data_args.eval_subset:
                knn_args.dstore_size = total_eval_tokens
                knn_args.dstore_damb = None
            # if knn_wrapper is not None:
            #     knn_wrapper.dstore_size, knn_wrapper.dstore_damb = knn_args.dstore_size,knn_args.dstore_damb
            #     knn_wrapper.break_into(model)
            if training_args.do_eval:
                do_eval()
            if training_args.do_predict:
                do_predict()

            if knn_args.build_index:
                knn_wrapper.build_index()

            if knn_args.cluster_dstore:
                knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
            
            if knn_wrapper is not None:
                knn_wrapper.break_out()
            dstore_path = knn_args.dstore_dir + '/0mmlu.pkl'
            with open(dstore_path, 'wb') as file:
                pickle.dump([knn_args.dstore_size,knn_args.dstore_damb], file)

if __name__ == "__main__":
    main()
