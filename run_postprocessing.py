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
import torch.nn as nn
from evaluate import load
import time
import string
import csv
import re
import pickle

import datasets
from datasets import load_dataset
from tqdm import tqdm

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
        '''if "validation" not in raw_datasets.keys():
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
            )'''
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

    # if not (training_args.do_train or data_args.eval_subset == 'train'):
    #     # If not training and not evaluating on train, we do not need to process it
    #     del raw_datasets["train"]
    #     del raw_datasets["test"]
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
    logger.info(f'Model assigned to {model.device}')
    
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)

    # model.to('cuda')
    # print("Model assigned to ",model.module.device)
    model.resize_token_embeddings(len(tokenizer))

    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    # print(model.forward.__code__.co_varnames)


    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def format_input(triviaq, text_column_name, prefix, sufix):
        # print(text_column_name, prefix, sufix)
        quest = triviaq["input_question"].replace("\"","'")
        if text_column_name == 'ip_question1':
            triviaq[text_column_name] = prefix + quest + sufix
        else:
            triviaq[text_column_name] = prefix + triviaq['answer1'] + sufix + quest
        return triviaq

    def tokenize_function_dstore(examples):
        # print("    ",examples[text_column_name][0])
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name][0])
            # print(output[0])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        # print(output.keys())
        return output

    def group_texts_question(examples):

        if len(examples) >= block_size:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is smaller than the length for the question"
            )

        input_ids = []
        attention_mask = []
        labels = []
        
        begin_loc = 0
        end_loc = min(len(examples['input_ids']), block_size)
        # if end_loc == block_size:
        #     print('yes')
        # else:
        #     print('no')
        cur_input_ids = examples['input_ids'][-end_loc:]
        cur_labels = list(cur_input_ids)
        cur_labels[:-end_loc] = [padding_index] * (len(cur_labels) - end_loc)
        
        attention_mask = [0] * (block_size - len(cur_input_ids)) + [1] * len(cur_labels)
        # print(len(cur_input_ids), block_size)

        if len(cur_input_ids) < block_size:
            padding_size = block_size - len(cur_input_ids)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad_ids = [pad_token_id] * padding_size
            pad_labels = [padding_index] * padding_size
               
            # cur_input_ids.append(input_ids)
            # attention_mask.append([1] * len(cur_labels))
            # labels.append(cur_labels)
            pad_ids += cur_input_ids
            pad_labels += cur_labels
            # print("~",len(pad_ids), block_size)
            # input_ids.append(pad_ids)
            # labels.append(pad_labels)
            result = {'input_ids': pad_ids, 'labels': pad_labels, 'attention_mask': attention_mask}
        else:
            # input_ids.append(cur_input_ids)
            # labels.append(cur_labels)
            result = {'input_ids': cur_input_ids, 'labels': cur_labels, 'attention_mask': attention_mask}
        return result

    # sampled_quests = raw_datasets[data_args.eval_subset].select(np.random.randint(0,raw_datasets[data_args.eval_subset].num_rows-10,4))
    prefix1 = """Answer these questions:
Q: Who was President when the first Peanuts cartoon was published?
A: The first Peanuts strip debuted on October 2, 1950. The president at the time was Harry S. Truman.
Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
A: The Nobel Prize in Literature was awarded to Sinclair Lewis in 1930.
Q: Where in England was Dame Judi Dench born?
A: Dame Judi Dench a star from York, England.
Q: In which decade did Billboard magazine first publish and American hit chart?
A: They published their first music 'hit parade' in 1936, which was the 30s.
Q: """
    
    """You are a robot that only outputs JSON.
You reply in JSON format with the field 'answer'.
Example question: Who was President when the first Peanuts cartoon was published? Example answer: {answer: Harry S. Truman}
Example question: """ 
        
    """Answer these questions:
Q: Who was President when the first Peanuts cartoon was published?
A: When the first Peanuts comic strip was published on October 2, 1950, Harry S. Truman was serving as President of the United States. The strip initially appeared in seven newspapers nationwide45, marking the beginning of what would become one of the most iconic comic strips in American history5.
Q: Where in England was Dame Judi Dench born?
A: Dame Judi Dench was born in Heworth, York, North Yorkshire, England on December 9, 1934110. Specifically, she was born in the York neighborhood of Heworth
Q: """
    sufix1 = """
A:"""
    # prompt_quests = "Answer these questions. "
    #"""Give your answer like an old timey private investigator hunting down a case step by step."""
    '''
    You are a robot that only outputs JSON.
You reply in JSON format with the field 'answer'.
Example question: Who was President when the first Peanuts cartoon was published? Example answer: {'answer': Harry Truman}
Now here is my question: 
'''
    # prompt_quests = "Answer these questions:\nQ: Who was President when the first Peanuts cartoon was published?\nA: Let's think step by step. The first Peanuts strip debuted on October 2, 1950. The president at the time was Harry S. Truman. That means the president was Harry S. Truman.\nQ: In which decade did Billboard magazine first publish and American hit chart?\nA: Let's think step by step. They published their first music 'hit parade' in 1936. That means the decade was 30s."
    #"Answer these questions:\nQ: The orrery, invented in 1710 by George Graham was in use for several centuries despite its inaccuracies in size and distance. What is it a model of?\nA: The orrery is a model of the Solar System.\nQ: How many stars are on the national flag of New Zealand?\nA: The number of start on the national flag of New Zealand are four.\nQ: Who produces the perfumes \"Opium\" and \"Rive Gauche\"?\nA: the perfumes Opium and Rive Gauche were produced by Yves Saint Laurent.\nQ: Which President of the USA has daughters named Malia Ann and Natasha or more famously Sasha?\nA: Malia Ann and Natasha are daughter of the US president BARACK OBAMA."

    # "Answer these questions:\nQ: The orrery, invented in 1710 by George Graham was in use for several centuries despite its inaccuracies in size and distance. What is it a model of?\nA: The orrery is a model of the Solar System.\nQ: How many stars are on the national flag of New Zealand?\nA: There are four stars on the national flag of New Zealand.\nQ: Who produces the perfumes 'Opium' and 'Rive Gauche'?\nA: Yves Saint Laurent produced the perfumes Opium and Rive Gauche.\nQ: Which President of the USA has daughters named Malia Ann and Natasha or more famously Sasha?\nA: Malia Ann and Natasha are daughters of the US president Barack Obama."
    # "\nA: Let's think step by step."
    # prompt_quests = "Answer these questions based on the hints:\nQ: The orrery, invented in 1710 by George Graham was in use for several centuries despite its inaccuracies in size and distance. What is it a model of?\nH: The orrery is a model of the Solar System.\nA: solar system\nQ: How many stars are on the national flag of New Zealand?\nH: There are four stars on the national flag of New Zealand.\nA: Four.\nQ: Who produces the perfumes \"Opium\" and \"Rive Gauche\"?\nH: The perfumes Opium and Rive Gauche were produced by Yves Saint Laurent.\nA: Yves Saint Laurent\nQ: Which President of the USA has daughters named Malia Ann and Natasha or more famously Sasha?\nH: Malia Ann and Natasha are daughters of the US president BARACK OBAMA.\nA: Barak Obama."
    # prompt_quests = "Answer these questions:\nQ: Who was President when the first Peanuts cartoon was published?\nQ: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\nQ: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nQ: Which American-born Sinclair won the Nobel Prize for Literature in 1930?\nA: Sinclair Lewis\nQ: Where in England was Dame Judi Dench born?\nQ: Where in England was Dame Judi Dench born?\nA: York\nQ: William Christensen of Madison, New Jersey, has claimed to have the world's biggest collection of what?\nQ: William Christensen of Madison, New Jersey, has claimed to have the world's biggest collection of what?\nA: Beer Cans\nQ: In which decade did Billboard magazine first publish and American hit chart?\nQ: In which decade did Billboard magazine first publish and American hit chart?\nA: 30s"
    column_names = raw_datasets[data_args.eval_subset].column_names
    text_column_name = 'input_final_prompts'
    # text_column_name = "ip_question2" if "ip_question1" in column_names else "ip_question1"
    # print(text_column_name)
    # with training_args.main_process_first(desc="dataset map input1"):
    #     raw_datasets = raw_datasets.map(
    #         format_input,
    #         fn_kwargs={
    #             "text_column_name": text_column_name,
    #             "prefix": prefix1,
    #             "sufix": sufix1
    #         },
    #         # batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc="dataset map context",
    #     )
    # print(raw_datasets[data_args.eval_subset][20])
    column_names = raw_datasets[data_args.eval_subset].column_names
    # print(raw_datasets[data_args.eval_subset]['question'][:5])
    # print(prompt_quests)

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

    if data_args.prompt:
        # seed = 89
        # sample_size = min(50, raw_datasets[data_args.eval_subset].num_rows)
        # shuffled_dataset = raw_datasets[data_args.eval_subset].shuffle(seed=seed)
        # sampled_dataset = shuffled_dataset.select(range(sample_size))
        # raw_datasets[data_args.eval_subset] = sampled_dataset
        # datastats = datasetStats()
        # datastats.stats(raw_datasets[data_args.eval_subset]["ip_question"], "sampleIP"+data_args.eval_subset+".png")
        # print(raw_datasets[data_args.eval_subset].num_rows)
        raw_datasets[data_args.eval_subset] = raw_datasets[data_args.eval_subset].select(range(0,50))
        # if training_args.do_train:
        # column_names = raw_datasets[data_args.eval_subset].column_names
        # text_column_name = "question" if "question" in column_names else column_names[0]
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function_dstore,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts_question,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping  texts in chunks of {block_size}",
            )
        # print(lm_datasets[data_args.eval_subset].column_names)

    for split, data in lm_datasets.items():
        total_eval_tokens = 0
        for chunk in data['labels']:
            total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
        logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')
        # if knn_args.dstore_size is None and split == 'train':
        # if knn_args.dstore_size is None and split == data_args.eval_subset:
        #     knn_args.dstore_size = total_eval_tokens
        print('Dstore size: ', knn_args.dstore_size)

    if training_args.do_eval or data_args.prompt:
        if data_args.eval_subset not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        # print("eval dataset picked with cols", eval_dataset.column_names)

    if knn_args.retomaton or knn_args.cluster_dstore:
        knn_wrapper = RetomatonWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda1=knn_args.lmbda1, lmbda2=knn_args.lmbda2, knn_temp=knn_args.knn_temp, probe=knn_args.probe,
            no_pointer=knn_args.no_pointer, min_knns=knn_args.min_knns, max_knns=knn_args.max_knns,
            members=knn_args.members)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda1=knn_args.lmbda1, lmbda2=knn_args.lmbda2, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_damb = knn_args.dstore_damb, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)
        if knn_args.dstore_size is not None:
            knn_wrapper.global_reconstruct_index, knn_wrapper.global_index = knn_wrapper.global_faiss()

    def partial_match(preds, labels):
        pm = 0
        for p,l in zip(preds, labels):
            if l.lower() in p.lower():
                pm+=1
        pm = pm*100/len(preds)
        return {'partial_match': pm}

    def postprocess_text(answer):
        # print(answer)
        if answer == '': 
            return answer
        answer = answer.split('{answer: ')
        # print(answer)
        if len(answer)>1:
            answer = answer[1].split('\n')[0]
            # print(answer)
            if answer == '': 
                return answer
            # print(answer)
            brace_index = answer.find('}')
            # print(brace_index)
            if brace_index!=-1:
                answer = answer[:brace_index]
        else:
            if 'answer:' in answer[0]:
                answer = answer[0].split('answer:')[1]
                answer = answer.split('\n')[0]
            else:
                answer = answer[0]
                answer = answer.split('\n')[0]
        answer = answer.strip()
        # print(answer)
        answer = answer.lower().replace('-',' ')
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        answer = answer.translate(repl_table)

        words = answer.split() 
        filtered_words = [word for word in words if word not in ['a', 'the', 'an']] 
        answer =  ' '.join(filtered_words)
        # print(answer)
        return answer
  
    def match(pred, answers):
        # answers = answers[2:-2].split("', '")
        # print(answers, pred)
        # pred = postprocess_text(pred)
        # print(pred)
        if pred in answers:
            return 1
        return 0

    def compute_metrics_alias(preds, labels):
        # result = bertscore.compute(predictions=preds, references=labels, model_type="distilbert-base-uncased")
        em = 0
        pm = 0 
        # references = np.char.lower(references)

        for pred, label in zip(preds, labels):
            if pred.strip() in label:
                em+=1
            for l in label:
                if l in pred:
                    pm+=1
                    break
            

        result = {"exact_match": em*100/len(preds)}
        print(f"exact_match: {result['exact_match']}")

        result["partial_match"] = pm*100/len(preds)
        print(f"Noisy answer: {result['partial_match']}")

        bleu = sacrebleu.compute(predictions=preds, references=labels)
        result["bleu"] = bleu["score"]
        print(f"bleu: {bleu['score']}")

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_metrics(pred, label):
        # result = bertscore.compute(predictions=preds, references=labels, model_type="distilbert-base-uncased")
        em = 0
        pm = 0 
        # references = np.char.lower(references)

        if pred.strip() in label:
            em = 1
        for l in label:
            if l in pred:
                pm = 1
                break
            
        result = {"exact_match": em}
        # print(f"exact_match: {result['exact_match']}")

        result["partial_match"] = pm
        # print(f"Noisy answer: {result['partial_match']}")
        # print(pred, label)
        bleu = sacrebleu.compute(predictions=[pred], references=[label])
        result["bleu"] = bleu["score"]
        # print(f"bleu: {bleu['score']}")

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def add_answer(example, ind, answer_list = None, match_list = None, output_list = None, first_pass = True):
        if first_pass:
            example['output1'] = output_list[ind]
            example['answer1'] = answer_list[ind]
        else:
            example['output2'] = output_list[ind]
            example['answer2'] = answer_list[ind]
            example['match'] = match_list[ind]
        return example


    if data_args.prompt:
        if not knn_args.knn and not knn_args.retomaton:
            knn_args.dstore_size = 0
        # print("*** Generating answers ***")
        # sacrebleu = load("sacrebleu")
        # exact_match = load("exact_match")
        # bertscore = load("bertscore")
        # true_answers = []
        '''norm_aliases = []
        max_alias = 0
        for qa in raw_datasets[data_args.eval_subset]:
            # true_answers.append(qa["answer"]["value"])
            norm_aliases.append(qa["answer"]["normalized_aliases"])
        # print(norm_aliases)
            if len(qa["answer"]["normalized_aliases"])>max_alias:
                max_alias = len(qa["answer"]["normalized_aliases"])
        print("Max aliases: ", max_alias)
        norm_aliases = [
        ref + [ref[0]] * (max_alias - len(ref))  # Duplicate first reference as padding
        for ref in norm_aliases]'''
        # print(model.forward)
        # print(eval_dataset.column_names)
        # generated_ids = model.generate(tokenizer.encode(data_args.prompt, return_tensors='pt').to(training_args.device), num_beams=5, num_return_sequences=5, do_sample=True)
        answers = []
        outputs = []
        if knn_args.knn:
            stride = 4
        else:
            stride = 4
        # print(eval_dataset.num_rows)
        max_new_tokens = 24
        knn_wrapper.num_beams = 5
        # start = time.time()
        if knn_args.knn:
            dstore_path = knn_args.dstore_dir + '/0file.pkl'
            with open(dstore_path, 'rb') as file:
                knn_wrapper.dstore_sizes = pickle.load(file)
        #[30719, 5119, 790, 5120, 19455, 34815, 2047, 2048, 5121, 13311, 4095, 17407, 1023, 9215, 6143, 18431, 15359, 27647, 2049, 3071, 15360, 46079, 3072, 591, 6144, 39935, 28671, 6145, 15361, 47103, 14335, 1024, 8191, 1025, 2050, 6146, 19456, 16383, 712, 13312, 35839, 33791, 10239, 4096, 2051, 4097, 3073, 6147, 5122, 11263]
        for i in tqdm(range(0, eval_dataset.num_rows, stride), desc = 'Generating open ended answer'):
            # if knn_args.knn:
                # knn_wrapper.dstore_size = dstore_sizes[i][0]
                # knn_wrapper.dstore_damb = dstore_sizes[i][1]
                # knn_wrapper.reconstruct_index, knn_wrapper.index = knn_wrapper.setup_faiss()
            
            # tokenizer.pad_token_id = tokenizer.eos_token_id
            input_ids = torch.tensor(eval_dataset["input_ids"][i:i+stride])
            attention_mask = torch.tensor(eval_dataset["attention_mask"][i:i+stride])
            # print(input_ids, attention_mask)

            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            # logger.info(f'Input_ids: {input_ids.shape}')
            # print(input_ids.shape, input_ids, attention_mask.shape, attention_mask)

            # print(f"model: {model.device}, data: {input_ids.device}, {attention_mask.device}")
            # print(input_ids)
            generated_ids = model.generate(inputs = input_ids, attention_mask = attention_mask, num_beams=knn_wrapper.num_beams, num_return_sequences=1, do_sample=True, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id)
            for generated_id in generated_ids:
                output = ""
                answer = ""
                # print(generated_id[:-1024])
                for ind, beam_output in enumerate(generated_id):

                    if ind < block_size:
                        continue
                    if beam_output == tokenizer.eos_token_id or beam_output == 128001:
                        continue
                    output+=tokenizer.decode(beam_output)
                    # logger.info(f'{i}: {tokenizer.decode(beam_output)}')
                # print(qa["question"]," True: ", qa["answer"]["value"]," pred: ", answer)
                # print('Before: ', answer)
                # answer = output.split('\n')[0]
                # print('After: ', answer)
                answers.append(output.split('\n')[0])
                outputs.append(output)
        # print(answers)

    if data_args.post_process:
        if knn_wrapper is not None:
            knn_wrapper.break_out()
        
        raw_datasets = raw_datasets.map(
             lambda example, ind: add_answer(example, ind, output_list = outputs, answer_list = answers, first_pass = True), with_indices=True 
        )

        prefix2 = """You are a robot that only outputs JSON.
You reply in JSON format with the field 'answer'.
Given the following information: The first Peanuts strip debuted on October 2, 1950. The president at the time was Harry S. Truman.
Example question: Who was President when the first Peanuts cartoon was published? Example answer: {answer: Harry S. Truman}
Given the following information: """ 
        sufix2 = """
Example question: """
        max_new_tokens = 12
        text_column_name = "ip_question2" if "ip_question1" in column_names else "ip_question1"
        # print(text_column_name)
        with training_args.main_process_first(desc="Generating extraction prompt"):
            raw_datasets = raw_datasets.map(
                format_input,
                fn_kwargs={
                "text_column_name": text_column_name,
                "prefix": prefix2,
                "sufix": sufix2
            },
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="dataset map context",
            )
        # print(raw_datasets[data_args.eval_subset]['ip_question2'])
        column_names = raw_datasets[data_args.eval_subset].column_names
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function_dstore,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        block_size = 512
        stride = 8
        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts_question,
                # batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping  texts in chunks of {block_size}",
            )
        
        outputs = []
        answers = []
        eval_dataset = lm_datasets[data_args.eval_subset]
        
        for i in tqdm(range(0, eval_dataset.num_rows, stride), desc = 'Extracting answer'):

            # tokenizer.pad_token_id = tokenizer.eos_token_id
            input_ids = torch.tensor(eval_dataset["input_ids"][i:i+stride])
            attention_mask = torch.tensor(eval_dataset["attention_mask"][i:i+stride])

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

            # print(f"model: {model.device}, data: {input_ids.device}, {attention_mask.device}")

            generated_ids = model.generate(inputs = input_ids, attention_mask = attention_mask, num_beams=5, num_return_sequences=1, do_sample=True, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id)
            for generated_id in generated_ids:
                output = ""
                answer = ""
                for ind, beam_output in enumerate(generated_id):
                    if ind < block_size:
                        continue
                    if beam_output == tokenizer.eos_token_id or beam_output == 128001:
                        continue
                    output+=tokenizer.decode(beam_output)
                    # logger.info(f'{i}: {tokenizer.decode(beam_output)}')
                # print(qa["question"]," True: ", qa["answer"]["value"]," pred: ", answer)
                # print('Before: ', answer)
                # answer = output.split('\n')[0]
                # print('After: ', answer)
                answers.append(postprocess_text(output))
                outputs.append(output)
        
        match_list = [match(pred, ans) for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['input_correct_responses'])]
        em = sum(match_list)/raw_datasets[data_args.eval_subset].num_rows*100
        # print(answers)
        raw_datasets = raw_datasets.map(
             lambda example, ind: add_answer(example, ind, output_list = outputs, answer_list = answers, match_list = match_list, first_pass = False), with_indices=True 
        )
        time = datetime.datetime.now()
        path = "output/output"+ time.strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
        with open(path, mode='w', newline='', encoding='utf-8') as file1:
            writer = csv.DictWriter(file1, fieldnames=['input1', 'output1', 'answer1', 'input2', 'output2', 'question', 'correct_answers', 'answer2', 'exact_match'])
            writer.writeheader()
            for row in raw_datasets[data_args.eval_subset]:
                # file.write("Input "+ qa['ip_question']+ 'Actual '+qa["question"]+" True: "+ qa["answer"]["value"]+" pred: "+ pred+"\n")
                # writer.writerow({'input_prompt': qa[text_column_name], 'question': qa['question'], 'correct_answers' : qa['answer']['normalized_aliases'], 'output_prediction' : op, 'processed_answer':ans, 'exact_match':metr['exact_match'], 'partial_match': metr['partial_match'], 'BLEU_score': metr['bleu']}) #, 'exact_match':metr['exact_match'], 'partial_match': metr['partial_match'], 'BLEU_score': metr['bleu']})
                writer.writerow({'input1': row['ip_question1'], 'output1': row['output1'], 'answer1' : row['answer1'], 'input2' : row['ip_question2'], 'output2' : row['output2'], 'question' : row['input_question'], 'correct_answers' : row['input_correct_responses'], 'answer2': row['answer2'], 'exact_match': row['match']})
        
        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
        
        fieldnames = ['time', 'model', 'dataset', 'retomaton', 'knn', 'lambda', 'block_size', 'k', 'knn_temp', 'max_new_tokens', 'dstore_size', 'exact_match', 'notes']
        metrics_to_file = { 'time': time_st,
            'model': model_args.model_name_or_path,
            'dataset': data_args.dataset_name,
            'retomaton': knn_args.retomaton, 
            'knn': knn_args.knn, 
            'lambda': knn_args.lmbda1, # + '+' + knn_args.lmbda2, 
            'block_size': block_size, 
            'k': knn_args.k, 
            'knn_temp': knn_args.knn_temp,
            'max_new_tokens' : max_new_tokens,
            'dstore_size' : 'local', #knn_args.dstore_size,
            'exact_match' : round(em, 4),
            'notes':  'entire dataset KNN model',
        }
        logger.info(metrics_to_file)
        fieldnames = list(metrics_to_file.keys())
        csv_file = "output/runs.csv"
        if not os.path.isfile(csv_file):
            # If it doesn't exist, write the dictionary to a new CSV file (with header)
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # Write header
                writer.writerow(metrics_to_file)  # Write data
        else:
            # If it exists, append the dictionary to the CSV file (without header)
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(metrics_to_file)  # Write data
    else:
        # print(answers)
        if knn_args.dstore_size is None:
            knn_args.dstore_size = 0
        answers = [postprocess_text(ans) for ans in answers]
        match_list = [match(pred, ans) for pred,ans in zip(answers, raw_datasets[data_args.eval_subset]['input_correct_responses'])]
        em = sum(match_list)/raw_datasets[data_args.eval_subset].num_rows*100
        # print(len(answers), len(match_list), raw_datasets['latest'].num_rows)
        raw_datasets = raw_datasets.map(
            lambda example, ind: add_answer(example, ind, output_list = outputs, answer_list = answers, match_list = match_list, first_pass = False), with_indices=True, desc = 'mapping output'
        )

        time = datetime.datetime.now()
        path = "output/output"+ time.strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
        with open(path, mode='w', newline='', encoding='utf-8') as file1:
            writer = csv.DictWriter(file1, fieldnames=['input1', 'output1', 'question', 'correct_answers', 'answer1', 'exact_match'])
            writer.writeheader()
            for row in raw_datasets[data_args.eval_subset]:
                # file.write("Input "+ qa['ip_question']+ 'Actual '+qa["question"]+" True: "+ qa["answer"]["value"]+" pred: "+ pred+"\n")
                # writer.writerow({'input_prompt': qa[text_column_name], 'question': qa['question'], 'correct_answers' : qa['answer']['normalized_aliases'], 'output_prediction' : op, 'processed_answer':ans, 'exact_match':metr['exact_match'], 'partial_match': metr['partial_match'], 'BLEU_score': metr['bleu']}) #, 'exact_match':metr['exact_match'], 'partial_match': metr['partial_match'], 'BLEU_score': metr['bleu']})
                writer.writerow({'input1': row['input_final_prompts'], 'output1': row['output2'], 'question' : row['input_question'], 'correct_answers' : row['input_correct_responses'], 'answer1': row['answer2'], 'exact_match': row['match']})
        
        time_st = time.strftime("%Y-%m-%d_%H:%M:%S")
        
        fieldnames = ['time', 'model', 'dataset', 'retomaton', 'knn', 'lambda', 'block_size', 'k', 'knn_temp', 'max_new_tokens', 'dstore_size', 'exact_match', 'notes']
        metrics_to_file = { 'time': time_st,
            'model': model_args.model_name_or_path,
            'dataset': data_args.dataset_name,
            'retomaton': knn_args.retomaton, 
            'knn': knn_args.knn, 
            'lambda': knn_args.lmbda1,# + '+' + knn_args.lmbda2, 
            'block_size': block_size, 
            'k': knn_args.k, 
            'knn_temp': knn_args.knn_temp,
            'max_new_tokens' : max_new_tokens,
            'dstore_size' : knn_args.dstore_size,
            'exact_match' : round(em, 2),
            'notes':  'direct answer generation',
        }
        logger.info(metrics_to_file)
        fieldnames = list(metrics_to_file.keys())
        csv_file = "output/runs.csv"
        if not os.path.isfile(csv_file):
            # If it doesn't exist, write the dictionary to a new CSV file (with header)
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # Write header
                writer.writerow(metrics_to_file)  # Write data
        else:
            # If it exists, append the dictionary to the CSV file (without header)
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(metrics_to_file)  # Write data

    if knn_args.build_index:
        knn_wrapper.build_index()

    if knn_args.cluster_dstore:
        knn_wrapper.cluster_dstore(num_clusters=knn_args.num_clusters, sample_size=knn_args.sample_size, model=model)
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()
    
if __name__ == "__main__":
    main()
