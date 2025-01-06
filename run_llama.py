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

from stats import datasetStats
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
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="checkpoints")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.25)
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
    print("Model assigned to ",model.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to('cuda')
    print("Model assigned to ",model.module.device)
    model.module.resize_token_embeddings(len(tokenizer))

    # Injecting KNN
    dimension = model.module.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    
    def extract_context(triviaq):
        if len(triviaq["entity_pages"]["wiki_context"])==0:
            # print("search_context")
            triviaq["context"] = triviaq["search_results"]["search_context"][0]
        else:
            triviaq["context"] = triviaq["entity_pages"]["wiki_context"][0]
        context = triviaq["context"].split(" ")
        context = " ".join(context[:256])
        # print(context, "\n ~~~~")
        triviaq["memory"] = context + triviaq["question"]
        triviaq["ip_question"] = prompt_quests+  "\nQ: " + triviaq["question"] + "\nA:"
        triviaq["ip_memory"] = context + triviaq["ip_question"]
        # print(len(triviaq["memory"]))
            # print("wiki_context")
        # print(triviaq["context"])
        return triviaq

    def tokenize_function_dstore(examples):
        # print("    ",examples[text_column_name][0])
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
            # print(output[0])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    def tokenize_function_question(examples):
        with CaptureLogger(tok_logger) as cl:
            if data_args.source_prefix is not None and data_args.source_sufix is not None:
                inputs = [data_args.source_prefix + inp + data_args.source_sufix for inp in examples[text_column_name]]
            elif data_args.source_prefix is not None :
                inputs = [data_args.source_prefix + inp for inp in examples[text_column_name]]
            elif data_args.source_sufix is not None :
                inputs = [inp + data_args.source_sufix for inp in examples[text_column_name]]
            else:
                inputs = [inp for inp in examples[text_column_name]]

            
            output = tokenizer(inputs)
            # print(output[0])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
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

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts_context(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
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
        for i in tqdm(range(0, total_length, data_args.stride), total=total_length):
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

    sampled_quests = raw_datasets[data_args.eval_subset].select(np.random.randint(0,raw_datasets[data_args.eval_subset].num_rows-10,4))
    prompt_quests = "Answer these questions:"
    for quest in sampled_quests:
        prompt_quests+="\nQ: "+quest["question"]

    with training_args.main_process_first(desc="dataset map context"):
        raw_datasets = raw_datasets.map(
            extract_context,
            # batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="dataset map context",
        )
    column_names = raw_datasets[data_args.eval_subset].column_names
    print(raw_datasets[data_args.eval_subset]['question'][:5])
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

    if knn_args.build_index or knn_args.save_knnlm_dstore or knn_args.cluster_dstore:
        # print(raw_datasets["validation"]["context"])
        
        datastats = datasetStats()
        datastats.stats(raw_datasets[data_args.eval_subset]["ip_question"], "wholeIP"+data_args.eval_subset+".png")
        seed = 89
        sample_size = min(1000, raw_datasets[data_args.eval_subset].num_rows)
        shuffled_dataset = raw_datasets[data_args.eval_subset].shuffle(seed=seed)
        sampled_dataset = shuffled_dataset.select(range(sample_size))
        raw_datasets[data_args.eval_subset] = sampled_dataset
        print(raw_datasets[data_args.eval_subset].num_rows)
        datastats.stats(raw_datasets[data_args.eval_subset]["ip_question"], "sampleIP"+data_args.eval_subset+".png")

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        # if training_args.do_train:
            # print()
        # else:
        #     column_names = raw_datasets["validation"].column_names
        text_column_name = "context" if "context" in column_names else column_names[0]

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function_dstore,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts_context,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
        )

    if data_args.prompt or training_args.do_eval:
        seed = 89
        sample_size = min(1000, raw_datasets[data_args.eval_subset].num_rows)
        shuffled_dataset = raw_datasets[data_args.eval_subset].shuffle(seed=seed)
        sampled_dataset = shuffled_dataset.select(range(sample_size))
        raw_datasets[data_args.eval_subset] = sampled_dataset
        datastats = datasetStats()
        datastats.stats(raw_datasets[data_args.eval_subset]["ip_question"], "sampleIP"+data_args.eval_subset+".png")
        # print(raw_datasets[data_args.eval_subset].num_rows)

        text_column_name = "ip_memory" if "ip_memory" in column_names else column_names[0]
        # if training_args.do_train:
        # column_names = raw_datasets[data_args.eval_subset].column_names
        # text_column_name = "question" if "question" in column_names else column_names[0]
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function_dstore,
                batched=True,
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
                desc=f"Grouping texts in chunks of {block_size}",
            )

    for split, data in lm_datasets.items():
        total_eval_tokens = 0        
        for chunk in data['labels']:
            total_eval_tokens += len([x for x in chunk[1:] if x != padding_index])
        logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')
        # if knn_args.dstore_size is None and split == 'train':
        if knn_args.dstore_size is None and split == data_args.eval_subset:
            knn_args.dstore_size = total_eval_tokens
    # print(knn_args.dstore_size)

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval or data_args.prompt:
        if data_args.eval_subset not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
    )

    if knn_args.retomaton or knn_args.cluster_dstore:
        knn_wrapper = RetomatonWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe,
            no_pointer=knn_args.no_pointer, min_knns=knn_args.min_knns, max_knns=knn_args.max_knns,
            members=knn_args.members)
    elif knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)

    # Training
    if training_args.do_train:
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

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        
        logger.info('Evaluation complete, calculating perplexity')

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        if knn_wrapper is not None:
            knn_metrics = knn_wrapper.get_metrics()
            metrics.update(knn_metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    sacrebleu = load("sacrebleu")
    exact_match = load("exact_match")
    # bertscore = load("bertscore")

    def partial_match(preds, labels):
        pm = 0
        for p,l in zip(preds, labels):
            if l.lower() in p.lower():
                pm+=1
        pm = pm*100/len(preds)
        return {'partial_match': pm}

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics_true(preds, labels):
        # preds, labels = eval_preds
        '''
        if isinstance(preds, tuple):
            preds = preds[0]
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id) #mask pad tokens
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) #now decode
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        '''
        # Some simple post-processing
        

        # result = bertscore.compute(predictions=preds, references=labels, model_type="distilbert-base-uncased")
        em = exact_match.compute(predictions = preds, references = labels, ignore_case = True, ignore_punctuation = True)
        result = {"exact_match": em["exact_match"]}
        print(f"exact_match: {em['exact_match']}")

        par_match = partial_match(preds, labels)
        result["partial_match"] = par_match["partial_match"]
        print(f"Noisy answer: {par_match['partial_match']}")

        processed_preds, processed_labels = postprocess_text(preds, labels)
        bleu = sacrebleu.compute(predictions=processed_preds, references=processed_labels)
        result["bleu"] = bleu["score"]
        print(f"bleu: {bleu['score']}")

        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_metrics_alias(preds, labels):
        # result = bertscore.compute(predictions=preds, references=labels, model_type="distilbert-base-uncased")
        em = 0
        pm = 0 
        preds = np.char.lower(preds)
        # references = np.char.lower(references)

        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        preds = np.char.translate(preds, table=repl_table)

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

    true_answers = []
    norm_aliases = []
    max_alias = 0
    for qa in raw_datasets[data_args.eval_subset]:
        true_answers.append(qa["answer"]["value"])
        norm_aliases.append(qa["answer"]["normalized_aliases"])
        if len(qa["answer"]["normalized_aliases"])>max_alias:
            max_alias = len(qa["answer"]["normalized_aliases"])
    print("Max length: ", max_alias)
    norm_aliases = [
    ref + [ref[0]] * (max_alias - len(ref))  # Duplicate first reference as padding
    for ref in norm_aliases]

    if data_args.prompt:
        # print(model.forward)
        # print(eval_dataset.column_names)
        # generated_ids = model.generate(tokenizer.encode(data_args.prompt, return_tensors='pt').to(training_args.device), num_beams=5, num_return_sequences=5, do_sample=True)
        answers = []
        outputs = []
        stride = 4
        print(eval_dataset.num_rows)
        max_new_tokens = 8
        # start = time.time()
        for i in tqdm(range(0, eval_dataset.num_rows, stride)):

            # tokenizer.pad_token_id = tokenizer.eos_token_id
            input_ids = torch.tensor(eval_dataset["input_ids"][i:i+stride])
            attention_mask = torch.tensor(eval_dataset["attention_mask"][i:i+stride])

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

            # print(f"model: {model.device}, data: {input_ids.device}, {attention_mask.device}")

            generated_ids = model.module.generate(inputs = input_ids, attention_mask = attention_mask, num_beams=5, num_return_sequences=1, do_sample=True, max_new_tokens = max_new_tokens, pad_token_id = tokenizer.eos_token_id)
            for generated_id in generated_ids:
                output = ""
                answer = ""
                for i, beam_output in enumerate(generated_id):
                    if i < block_size:
                        continue
                    output+=tokenizer.decode(beam_output)
                    # logger.info(f'{i}: {tokenizer.decode(beam_output)}')
                # print(qa["question"]," True: ", qa["answer"]["value"]," pred: ", answer)
                # print('Before: ', answer)
                answer = output.split('\n')[0]
                # print('After: ', answer)
                answers.append(answer)
                outputs.append(output)
            # print(answers)
                # print(answer)
        # logger.info('Generation took {} s'.format(time.time() - start))
        start = time.time()
        metrics = compute_metrics_true(answers, true_answers)
        metrics_norm_alias = compute_metrics_alias(answers, norm_aliases)
        logger.info('Metrics calculation took {} s'.format(time.time() - start))
        path = "outputs/output"+ datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
        # file = open(path, "w")
        # file.write(f"retomaton: {knn_args.retomaton} knn: {knn_args.knn} lambda: {knn_args.lmbda} block_size: {block_size}, k: {knn_args.k}, knn_temp: {knn_args.knn_temp}\n")
        # file.write(f"Metrics with answer: {metrics}\n")
        # file.write(f"Metrics with aliases: {metrics_norm_alias}\n")
        # file.write(f"Input format: {data_args.source_prefix} <question> {data_args.source_sufix}\n\n")
        with open(path, mode='w', newline='', encoding='utf-8') as file1:
            writer = csv.DictWriter(file1, fieldnames=['input_prompt', 'question', 'correct_answers', 'output_prediction', 'processed_answer'])
            writer.writeheader()
            for ans, op, qa in zip(answers, outputs, raw_datasets[data_args.eval_subset]):
                # file.write("Input "+ qa['ip_question']+ 'Actual '+qa["question"]+" True: "+ qa["answer"]["value"]+" pred: "+ pred+"\n")
                writer.writerow({'input_prompt': qa['ip_question'], 'question': qa['question'], 'correct_answers' : qa['answer']['normalized_aliases'], 'output_prediction' : op, 'processed_answer':ans})
                # generated_answers = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

                # Print or collect answers
                # print(generated_answers)
        # file.close()
        time_st = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        fieldnames = ['time', 'model', 'retomaton', 'knn', 'lambda', 'block_size', 'k', 'knn_temp', 'max_new_tokens', 'exact_match', 'partial_match', 'bleu', 'notes']
        metrics_to_file = { 'time': time_st,
            'model': model_args.model_name_or_path,
            'retomaton': knn_args.retomaton, 
            'knn': knn_args.knn, 
            'lambda': knn_args.lmbda, 
            'block_size': block_size, 
            'k': knn_args.k, 
            'knn_temp': knn_args.knn_temp,
            'max_new_tokens' : max_new_tokens
        }
        for key, val in metrics_norm_alias.items():
            metrics_to_file[key] = val
        metrics_to_file['notes'] = 'Input: ip_memory'
        fieldnames = list(metrics_to_file.keys())
        csv_file = "outputs/metrics.csv"
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
