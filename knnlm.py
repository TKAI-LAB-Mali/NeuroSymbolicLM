import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import gc

import faiss
import faiss.contrib.torch_utils

logger = logging.getLogger(__name__)
logger.setLevel(20)

class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(object):
    def __init__(self, dstore_size, dstore_damb, dstore_dir, dimension, 
            knn_sim_func=None, knn_keytype=None,
            no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
            recompute_dists = False,
            k=512, t=10, lmbda1=0.25, lmbda2=0.1, knn_temp=1.0, probe=32):
        self.dstore_size = dstore_size
        self.dstore_damb = dstore_damb
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.k = k
        self.t = t
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None

        self.global_keys = None
        self.global_vals = None

        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        self.curr_tokens = 0
        self.unique = False

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func] # l2 or dot product function

    def cleanup_dstore(self):
        # logger.info(f'cleanup previous dstores')
        for attr in ['keys', 'vals']:
            if hasattr(self, attr):
                delattr(self, attr)

        torch.cuda.empty_cache()
        gc.collect()

    def global_faiss(self):
        logger.info(f'Loading global dstore')
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, None) 
        logger.info(f'Dstore path: {index_name}')
        cpu_global_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_global_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            logger.info(f'Moving to GPU')
            gpu_global_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_global_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_global_index = cpu_global_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_global_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb)
        if self.no_load_keys:
            self.global_keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.global_vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)
        # print(self.global_keys.shape)
        # print(self.global_vals.shape)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if self.no_load_keys:
                del self.global_keys
                self.global_keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                # print(self.global_keys_from_memmap.dtype, self.global_keys_from_memmap.shape)
                # self.global_keys = self.global_keys_from_memmap[:].astype(np.float16)
                

            del self.global_vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            self.global_vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_global_index, gpu_global_index

    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        self.cleanup_dstore()

        start = time.time()
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb) 
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        logger.info(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe

        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            logger.info(f'Moving to GPU')
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        if self.dstore_size >= pow(2, 14):
            cpu_index.make_direct_map()

        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb)
        keys = None
        if not self.no_load_keys:
            keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            # logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del keys
                keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            # print(f'vals {vals}')
            del vals_from_memmap
            # logger.info('Loading to memory took {} s'.format(time.time() - start))
        # gpu_id = 0
        # free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

        # print(f"Free memory: {free_mem / 1024 ** 2:.2f} MB")
        # print(f"Total memory: {total_mem / 1024 ** 2:.2f} MB")
        return gpu_index, keys, vals

    def break_into(self, model):
        logger.info(f'Registering hooks')
        self.model = model
        model.broken_into = True
        # self.reconstruct_index, self.index = self.setup_faiss()
        # self.global_reconstruct_index, self.global_index = self.global_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries, store):
        old_k = 0
        if self.k > self.dstore_size:
            old_k = self.k
            while self.k > self.dstore_size :
                self.k = self.k//2
        if not self.knn_gpu:
            queries = queries.cpu()
        if store:
            dists, knns = self.index.search(queries, self.k)
        else:
            # print(queries.shape)
            dists, knns = self.global_index.search(queries, self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        if old_k !=0:
            self.k = old_k
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)
    
    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask] # (nonpad, dim)

        if self.unique:
            dists, knns = self.get_knns(queries, True) # (nonpad batch * time, k)
            if self.recompute_dists:
                knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
                dists = self.dist_func(queries, knns_vecs) 
            
            neg_dists = -dists
            knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists, True)
            
            interpolated_scores = KNNWrapper.interpolate(knn_log_probs, None, lm_logits, self.lmbda1, self.lmbda2) # (nonpad, vocab)
            output[nonpad_mask] = interpolated_scores
            return output 
    
        if self.curr_tokens == 0:
            self.curr_tokens = self.max_new_tokens

        if self.curr_tokens == self.max_new_tokens:
            self.faiss_objects = []
            self.dstore_size_disamb = []
            for i in range(queries.shape[0]//self.num_beams):
                self.dstore_size, self.dstore_damb = self.dstore_sizes.pop(0)
                self.faiss_objects.append(self.setup_faiss())
                self.dstore_size_disamb.append((self.dstore_size, self.dstore_damb))
            self.curr_tokens -= 1
        else:
            self.curr_tokens -= 1

        unbatch_queries = torch.chunk(queries, queries.shape[0]//self.num_beams)
        knn_log_probs_list = []
        for que, fais_obj, dsa in zip(unbatch_queries, self.faiss_objects, self.dstore_size_disamb):
            self.dstore_size, self.dstore_damb = dsa
            self.index, self.keys, self.vals = fais_obj

            dists, knns = self.get_knns(que, True) # (nonpad batch * time, k)
            
            if self.recompute_dists:
                knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
                dists = self.dist_func(que, knns_vecs) 
        
            neg_dists = -dists
            knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists, True)
            knn_log_probs_list.append(knn_log_probs)

        if self.global_vals is not None:
            self.dstore_size, self.dstore_damb = self.global_dstore_size, None
            global_dists, global_knns = self.get_knns(queries, False)
            # logger.info(f'global_knns: {global_knns}')

            if self.recompute_dists:
                global_knns_vecs = torch.from_numpy(self.global_keys[global_knns]).to(self.device)
                global_dists = self.dist_func(queries, global_knns_vecs) 
            
            global_neg_dists = -global_dists
            global_knn_log_probs, _ = self.knns_to_log_prob(global_knns, global_neg_dists, False)
            # logger.info(f'global_knn_log_probs shape: {global_knn_log_probs.shape}')
            self.global_keys = self.global_keys_from_memmap[global_knns.cpu()].astype(np.float16)
            self.global_keys = torch.from_numpy(self.global_keys).long().to(self.device)
            # logger.info(f'global keys: {self.global_keys}')
            global_score = KNNWrapper.knn_relevance(queries, self.global_keys, self.t)
            # logger.info(f'global score: {global_score}')

            # logger.info(f'local score: {local_score}\nglobal score: {global_score}')
            # dist = min(dists)
            # global_dist = min(global_dists)
            # logger.info(f'local max:{torch.max(torch.unique(local_score, dim=0))} global_max:{torch.max(torch.unique(global_score, dim=0))}')
            unbatch_global_knn_log_probs = torch.chunk(global_knn_log_probs, global_knn_log_probs.shape[0]//self.num_beams)
            # logger.info(f'unbatch_global_knn_log_probs len: {len(unbatch_global_knn_log_probs)}')
            unbatch_global_score = torch.chunk(global_score, global_score.shape[0]//self.num_beams)
            interpolated_scores_list = []
            for l_k_probs, g_k_probs, logits, l_score, g_score in zip(knn_log_probs_list, unbatch_global_knn_log_probs, torch.chunk(lm_logits, lm_logits.shape[0]//self.num_beams), local_score_list, unbatch_global_score):
                # logger.info(f'local score shape: {l_score.shape}')
                # logger.info(f'global score shape: {g_score.shape}')
                local_max = torch.max(l_score)
                global_max = torch.max(g_score)
                if local_max>global_max:
                    # print('yes')
                    interpolated_score = KNNWrapper.interpolate(l_k_probs, g_k_probs, logits, self.lmbda1, self.lmbda2)
                else:
                    # print('no')
                    interpolated_score = KNNWrapper.interpolate(l_k_probs, g_k_probs, logits, self.lmbda2, self.lmbda1)
                interpolated_scores_list.append(interpolated_score)
            interpolated_scores = torch.cat(interpolated_scores_list)
            del self.global_keys
        else:
            interpolated_scores = KNNWrapper.interpolate(torch.cat(knn_log_probs_list), None, lm_logits, self.lmbda1, self.lmbda2) # (nonpad, vocab)
        output[nonpad_mask] = interpolated_scores
        return output 
    
    '''
    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.max_new_tokens == 0:
            self.max_new_tokens = 24
            del self.faiss_objects, self.dstore_size_disamb, self.index, self.keys, self.keys_from_memmap

        if self.max_new_tokens == 24:
            # self.dstore_damb = self.dstore_sizes[i][1]
            self.faiss_objects = []
            self.dstore_size_disamb = []
            for i in range(queries.shape[0]//self.num_beams):
                logger.info(f'Dstores left {len(self.dstore_sizes)}')
                self.dstore_size, self.dstore_damb = self.dstore_sizes.pop(0)
                # reconstruct_index, index = self.setup_faiss()
                self.faiss_objects.append(self.setup_faiss())
                self.dstore_size_disamb.append((self.dstore_size, self.dstore_damb))
            # logger.info(f'Dstores sizes {self.dstore_size_disamb}')
            self.max_new_tokens -= 1
        else:
            self.max_new_tokens -= 1
        # logger.info(f'tokens left to generate: {self.max_new_tokens}')
        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        # logger.info(f'lm_logits shape: {lm_logits.shape}')
        queries = queries[nonpad_mask] # (nonpad, dim)
        # logger.info(f'querries: {queries}')
        # logger.info(f'querries shape: {queries.shape}')
        unbatch_queries = torch.chunk(queries, queries.shape[0]//self.num_beams)
        # logger.info(f'unbatch_queries len: {len(unbatch_queries)}')
        interpolated_scores_list, knn_log_probs_list = [], []
        for que, fais_obj, dsa, logits in zip(unbatch_queries, self.faiss_objects, self.dstore_size_disamb, torch.chunk(lm_logits, lm_logits.shape[0]//self.num_beams)):
            # logger.info(f'unbatch_queries: {unbatch_queries}')
            self.dstore_size, self.dstore_damb = dsa
            self.index, self.keys_from_memmap, self.vals = fais_obj

            dists, knns = self.get_knns(que, True) # (nonpad batch * time, k)
            
            if self.recompute_dists:
                knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
                dists = self.dist_func(que, knns_vecs) 
        
            neg_dists = -dists
            knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists, True)
            # logger.info(knns.device)
            knn_log_probs_list.append(knn_log_probs)

            if self.global_dstore_size is not None:
                self.keys = self.keys_from_memmap[knns.cpu()].astype(np.float16)
                self.keys = torch.from_numpy(self.keys).long().to(self.device)
                
                local_score = KNNWrapper.knn_relevance(que, self.keys, self.t)

                self.dstore_size, self.dstore_damb = self.global_dstore_size, None
                global_dists, global_knns = self.get_knns(que, False)

                if self.recompute_dists:
                    global_knns_vecs = torch.from_numpy(self.global_keys[global_knns]).to(self.device)
                    global_dists = self.dist_func(queries, global_knns_vecs) 
                
                global_neg_dists = -global_dists
                global_knn_log_probs, _ = self.knns_to_log_prob(global_knns, global_neg_dists, False)
                # logger.info(f'global_knn_log_probs shape: {global_knn_log_probs.shape}')
                # logger.info(f'logits shape: {logits.shape}')
                self.global_keys = self.global_keys_from_memmap[global_knns.cpu()].astype(np.float16)
                self.global_keys = torch.from_numpy(self.global_keys).long().to(self.device)
                # logger.info(f'global keys: {self.global_keys}')
                global_score = KNNWrapper.knn_relevance(que, self.global_keys, self.t)
                local_max = torch.max(local_score)
                global_max = torch.max(global_score)
                if local_max>global_max:
                    # print('yes')
                    interpolated_score = KNNWrapper.interpolate(knn_log_probs, global_knn_log_probs, logits, self.lmbda1, self.lmbda2)
                else:
                    # print('no')
                    interpolated_score = KNNWrapper.interpolate(knn_log_probs, global_knn_log_probs, logits, self.lmbda2, self.lmbda1)
                interpolated_scores_list.append(interpolated_score)
                del self.global_keys
        if self.global_dstore_size is not None:
            interpolated_scores = torch.cat(interpolated_scores_list)
        else:
            interpolated_scores = KNNWrapper.interpolate(torch.cat(knn_log_probs_list), None, lm_logits, self.lmbda1, self.lmbda2) # (nonpad, vocab)
        output[nonpad_mask] = interpolated_scores
        return output
    '''
    def knns_to_log_prob(self, knns, neg_dists, local):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        if local:
            vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
            # print(f'knns: {knns}')
            # print(f'vals: {self.vals}')
            # print(f'vals_at_knns: {vals_at_knns}')
        else:
            vals_at_knns = self.global_vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns
        
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        logger.info(f'Removing hooks')
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
    
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def knn_relevance(query, keys, T):
        # distance = np.linalg.norm(query_embedding - doc_embedding)
        # return np.exp(-distance / T)
        distance = torch.norm(query.unsqueeze(1) - keys, dim = -1)
        return torch.exp(-distance / T)


    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)


    @staticmethod
    def interpolate(knn_log_probs, global_knn_log_probs, lm_log_probs, lmbda1, lmbda2):
        if global_knn_log_probs is not None:
            # print('global')
            interpolated = torch.logaddexp(lm_log_probs + np.log(1 - lmbda1 - lmbda2), torch.logaddexp( 
                knn_log_probs + np.log(lmbda1),
                global_knn_log_probs + np.log(lmbda2)))
        else:
            # print('local')
            interpolated = torch.logaddexp(
                lm_log_probs + np.log(1 - lmbda1), 
                knn_log_probs + np.log(lmbda1))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'gemma3_text' : {
            KEY_TYPE.last_ffn_input: (lambda model: model.model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.model.layers[-1], False), 
        },
        'llama': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.layers[-1], False), 
        },
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        }
}
    

class KNNSaver(object):
    def __init__(self, dstore_size, dstore_damb, dstore_dir, dimension, knn_keytype=None):
        self.dstore_size = dstore_size
        self.dstore_damb = dstore_damb
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        logger.info(f'Registering hooks')
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb)
        keys_filename = f'{keys_vals_prefix}_keys.npy'
        vals_filename = f'{keys_vals_prefix}_vals.npy'
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]
        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex

        self.dstore_idx += batch_time_size
        
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        logger.info(f'Removing hooks')
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000, 
            ncentroids=256, seed=1, code_size=64, probe=32):
        ncentroids = 256
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb) 
        # print(self.dstore_size, ncentroids)
        while self.dstore_size < ncentroids:
            ncentroids = ncentroids//2
        # logger.info(f'ncentroids: {ncentroids}')
        num_points = 9984

        if self.dstore_size < pow(2,14):
            logger.info(f'ncentroids: {ncentroids}')
            start = 0
            start_time = time.time()
            index = faiss.IndexFlatL2(self.dimension)
            index.add(torch.tensor(self.dstore_keys.astype(np.float32)))
        else:
            # Initialize faiss index
            if self.dstore_size > 1000000:
                ncentroids = 2048
                num_points = 100000
            logger.info(f'ncentroids: {ncentroids}')
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension,
                ncentroids, code_size, 8)
            index.nprobe = probe

            logger.info('Training Index')
            np.random.seed(seed)
            random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]), size=[min(num_points, self.dstore_vals.shape[0])], replace=False)
            # print(len(random_sample))
            start = time.time()
            # Faiss does not handle adding keys in fp16 as of writing this.
            index.train(self.dstore_keys[random_sample].astype(np.float32))
            logger.info(f'Training took {time.time() - start} s')

            logger.info('Adding Keys')
            # index = faiss.read_index(f'{index_name}.trained')
            start = 0
            start_time = time.time()
            while start < self.dstore_size:
                end = min(self.dstore_size, start + num_keys_to_add_at_a_time)
                to_add = self.dstore_keys[start:end].copy()
                index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
                start += num_keys_to_add_at_a_time

                if (start % 1000000) == 0:
                    logger.info(f'Added {start} tokens so far')
                    logger.info(f'Writing Index {start}')
                    faiss.write_index(index, f'{index_name}')

            logger.info(f'Adding total {start} keys')
        logger.info(f'Adding took {time.time() - start_time} s')
        logger.info(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        logger.info(f'Writing index took {time.time() - start} s')
        
    def get_metrics(self):
        return {}

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension, dstore_damb):
    if dstore_damb is None:
        return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'
    else:
        return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}_{dstore_damb}'

def get_index_path(dstore_dir, model_type, dstore_size, dimension, dstore_damb):
    if dstore_damb is None:
        return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'
    else:
        return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}_{dstore_damb}.indexed'