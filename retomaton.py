from collections import defaultdict
import os

import logging
import pickle
import time
import numpy as np
import torch
from transformers.generation import utils
from torch import nn
from enum import Enum, auto
from pathlib import Path
import glob
from itertools import zip_longest

from tqdm import tqdm

from knnlm import KNNWrapper, get_dstore_path

import faiss
import faiss.contrib.torch_utils
from faiss import IndexFlatL2
import scipy.sparse as sp


from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(20)

class RetomatonWrapper(KNNWrapper):
    def __init__(self, no_pointer=False, min_knns=1, max_knns=1024, members=None, **kwargs):
        super().__init__(**kwargs)
        self.no_pointer = no_pointer
        self.min_knns = min_knns
        self.max_knns = max_knns
        # self.load_retomaton(members)
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')

    def load_retomaton(self, members=None):
        if members is None:
            if self.dstore_damb is None:
                available_member_files = glob.glob(f'{self.dstore_dir}/members_*_{self.dstore_size}_{self.dimension}*')
            else:
                available_member_files = glob.glob(f'{self.dstore_dir}/members_*_{self.dstore_size}_{self.dimension}_{self.dstore_damb}*')
            if len(available_member_files) == 0:
                logger.info(f'No member files found in {self.dstore_dir}, not using clustering')
            else:
                members = available_member_files[0]
                logger.info(f'Found the following cluster members files: {available_member_files}')
                logger.info(f'Using members file {members}')
        
        if members is None:
            self.extend_pointers_using_clusters = lambda pointers: pointers
        else:
            with open(members, 'rb') as file:
                self.members = pickle.load(file)
            members_for_indices = np.nonzero(self.members[np.arange(self.members.shape[0])])
            self.cluster = torch.zeros((self.dstore_size, ), dtype=torch.int32).to(self.device)
            self.cluster[members_for_indices[1]] = torch.from_numpy(members_for_indices[0]).to(self.device)
        self.generate_cur_knns = None
        self.no_lookup_counter = 0
        self.no_lookup_counter_history = []

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            self.labels = input_ids.to(self.device)
        else:
            self.labels = labels.to(self.device)       
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1

        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        queries = self.activation_capturer.captured.to(self.device) # (batch, time, dim) 

        if self.labels.shape[1]>1:
            # New batch for generation. Emptying the pointers and neighbors
            self.generate_cur_knns = [] #torch.tensor([], dtype=torch.int64)
        else:
            order = utils.get_retoMaton_beam_idx()
            reordered_curr_knns = [self.generate_cur_knns[i] for i in order]
            reordered_vals_at_knns = [self.all_vals_at_knns[i] for i in order]
            self.generate_cur_knns = reordered_curr_knns
            self.all_vals_at_knns =  reordered_vals_at_knns
            for idx, (lab, vals) in enumerate(zip(self.labels, self.all_vals_at_knns)):
                knns = self.generate_cur_knns[idx]
                vals_are_correct_and_pointer_available = (vals == lab) & (knns < self.dstore_size - 1)
                self.generate_cur_knns[idx] = knns[vals_are_correct_and_pointer_available]
             
        nonpad_mask = torch.cat([
            torch.zeros([batch, time_dim - 1], dtype=torch.bool),
            torch.ones([batch, 1], dtype=torch.bool),
        ], axis=-1).to(self.device)

        captured_labels = None
        queries = queries[nonpad_mask] # (nonpad, dim)
        lm_logits = lm_logits[nonpad_mask] # (nonpad, vocab)

        all_knn_probs = []
        self.all_vals_at_knns = []
        
        for idx, timestep_query in enumerate(queries):
            perform_search = False
            extended_pointers = None
            if len(self.generate_cur_knns)<queries.shape[0]:
                cur_knns = torch.tensor([], dtype=torch.int64)
            else:
                temp_knns = self.generate_cur_knns[idx]
                valid_mask = temp_knns != -1
                cur_knns = temp_knns[valid_mask]
            pointers = cur_knns + 1
            if (pointers == self.dstore_size).any():
                pointers[pointers == self.dstore_size] = self.dstore_size-1
            if self.no_pointer or cur_knns.numel() < self.min_knns:
                perform_search = True
                self.no_lookup_counter_history.append(self.no_lookup_counter)
                self.no_lookup_counter = 0
            else:
                self.no_lookup_counter += 1

            if self.no_pointer:
                extended_pointers = None
            elif pointers.numel() >= self.max_knns:
                extended_pointers = pointers[:self.max_knns]
            else:
                extended_pointers = self.extend_pointers_using_clusters(pointers)
            
            # (vocab_size, ) , (k, ), (k, ), (k, )
            cur_knn_log_prob, knns, dists, vals_at_knns = self.get_knn_log_prob(
                timestep_query, 
                pointers=extended_pointers,
                perform_search=perform_search)
            # logger.info(f'Beam {idx}: knns: {vals_at_knns}\n val indices: {knns}')
            # logger.info(f'Neighbour: {self.tokenizer.decode((self.vals[knns[0]-10:knns[0]+1].squeeze()))}')
            all_knn_probs.append(cur_knn_log_prob)
            self.all_vals_at_knns.append(vals_at_knns)
            
            if not self.no_pointer:
                cur_knns = knns[dists.argsort(descending=True)].unsqueeze(0)
                if len(self.generate_cur_knns) < queries.shape[0]:
                    self.generate_cur_knns.append(cur_knns)
                else:
                    self.generate_cur_knns[idx] = cur_knns
            # logger.info(f'Beam {idx} Neighbour: {self.tokenizer.decode((self.vals[cur_knns.squeeze(0)[0]-10:cur_knns.squeeze(0)[0]].squeeze()))}"{self.tokenizer.decode((self.vals[cur_knns.squeeze(0)[0]]))}"')
            # try:
            #     logger.info(f'Beam {idx} Neighbour: {self.tokenizer.decode((self.vals[cur_knns.squeeze(0)[1]-10:cur_knns.squeeze(0)[1]].squeeze()))}"{self.tokenizer.decode((self.vals[cur_knns.squeeze(0)[1]]))}"')
            # except IndexError:
            #     logger.info('')

        interpolated_scores = KNNWrapper.interpolate(torch.stack(all_knn_probs), lm_logits, self.lmbda1) # (nonpad, vocab)
        output[nonpad_mask] = interpolated_scores
        del pointers
        torch.cuda.empty_cache()
        return output

    def get_knn_log_prob(self, query, pointers, perform_search):
        pointer_dists = torch.tensor([[]]).to(self.device)
        if pointers is not None and pointers.numel() > 0 and not self.recompute_dists:
            pointer_vectors = self.reconstruct_ids(pointers)
            pointer_dists = self.dist_func(query, pointer_vectors)
        
        if perform_search:
            dists, knns = self.get_knns(query.unsqueeze(0)) # (1, k)
            dists, knns = dists.squeeze(0), knns.squeeze(0) # (k, )
            if pointers is not None and pointers.numel() > 0:
                knns = torch.cat([knns, pointers], axis=-1)
                dists = torch.cat([dists, pointer_dists], axis=-1)
        else:
            knns = pointers
            dists = pointer_dists

        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(query, knns_vecs) 
        
        neg_dists = -dists       
        knn_log_probs, vals_at_knns = self.knns_to_log_prob(knns, neg_dists)
        
        return knn_log_probs, knns, neg_dists, vals_at_knns

    def extend_pointers_using_clusters(self, pointers):
        # logger.info(f'Extending pointers')
        if pointers.numel() == 0:
            return pointers
        # Don't take the same cluster twice
        # pointers = pointers.numpy()
        clusters, cluster_counts = torch.unique(self.cluster[pointers], return_counts=True)
        # Take smaller clusters first
        clusters = clusters[torch.argsort(-cluster_counts)]
        members = torch.from_numpy(np.nonzero(self.members[clusters.cpu().numpy()])[1]).to(self.device)
        # Prefer datastore entries that were directly pointed to by the previous time step's
        # datastore entries, over other members of their cluster
        extended_pointers = torch.cat([pointers, members])
        if len(extended_pointers) > self.max_knns:
            extended_pointers = extended_pointers[:self.max_knns]
        return extended_pointers

    def reconstruct_ids(self, ids):
        # logger.info(f'Reconstructing keys')
        # Converting to numpy only because GPU indexes do not support reconstructing vectors:
        # https://github.com/facebookresearch/faiss/issues/2181
        ids = ids.cpu().numpy()
        # faiss's index.reconstruct supports a single ID at a time, so batching is performed
        # using numpy.vectorize:
        # https://github.com/facebookresearch/faiss/issues/1163
        reconstruct_func = np.vectorize(lambda x: self.reconstruct_index.reconstruct(int(x)), otypes=[object])
        vectors = reconstruct_func(ids)
        vectors = np.stack(vectors).reshape(ids.shape + (self.dimension, ))
        vectors_t = torch.from_numpy(vectors).to(self.device)
        return vectors_t

    def get_metrics(self):
        return {'lookups_saved': np.sum(self.no_lookup_counter_history)/
            (np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history)),
        }

    def break_out(self):
        super().break_out()
        self.print_stats()
    
    def print_stats(self):
        if len(self.no_lookup_counter_history) > 0:
            saved = np.sum(self.no_lookup_counter_history) / \
                (np.sum(self.no_lookup_counter_history) + len(self.no_lookup_counter_history))
            logger.info(f'Lookups saved: {saved*100}%')

    def cluster_dstore(self, num_clusters, sample_size, model, batch_size=500000):
        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension, self.dstore_damb)
        keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
        # vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',shape=(self.dstore_size, 1))

        if sample_size > self.dstore_size:
            logger.info('Taking all data for training')
            to_cluster = keys[:]
        else:
            idx = np.random.RandomState(1).choice(np.arange(self.dstore_size), size=sample_size, replace=False)
            to_cluster = keys[idx]
        # print(vals[:])
        to_cluster = to_cluster.astype(np.float32)
        logger.info(f'Starting to cluster {sample_size} examples into {num_clusters} clusters')
        kmeans = faiss.Kmeans(self.dimension, num_clusters, niter=20, verbose=True, gpu=True, seed=1)
        kmeans.train(to_cluster)
        # print(kmeans.centroids)
        logger.info(f'Done training, assigning {self.dstore_size} examples to the clusters')

        index = IndexFlatL2(self.dimension)
        index.add(kmeans.centroids)
        print(len(kmeans.centroids))
        logger.info('Index created, added centroids')
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            logger.info('Moving index to GPU')
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, co)
            logger.info('Moved index to GPU')

        start = 0
        centroid_ids = []
        logger.info('Starting to add tokens')
        while start < self.dstore_size:
            end = min(self.dstore_size, start + batch_size)
            to_search = keys[start:end].copy()
            _, key_i = index.search(torch.from_numpy(to_search).float(), 1)
            centroid_ids.append(key_i.squeeze())
            start += batch_size
            if (start % 1000000) == 0:
                print('Assigned %d tokens so far' % start)

        centroid_ids = np.concatenate(centroid_ids)
        print(centroid_ids)
        logger.info('Processing the mapping of cluster->members')
        parent_cluster = centroid_ids
        cluster_to_members = defaultdict(set)
        for key_i, cluster in tqdm(enumerate(parent_cluster), total=self.dstore_size):
            cluster_to_members[cluster.item()].add(key_i)

        if len(cluster_to_members)>1:
            row_ind = [k for k, v in cluster_to_members.items() for _ in range(len(v))]
            col_ind = [i for ids in cluster_to_members.values() for i in ids]
            members_sp = sp.csr_matrix(([1]*len(row_ind), (row_ind, col_ind)))

            members_filename = get_members_path(self.dstore_dir, 
                model.config.model_type, self.dstore_size, self.dimension,
                sample_size, num_clusters,self.dstore_damb)
            with open(members_filename, 'wb') as f:
                pickle.dump(members_sp, f)

            logger.info(f'Done, found {len(cluster_to_members)} clusters, written to {members_filename}')
        else:
            logger.info(f'Only found {len(cluster_to_members)} clusters skipping clustering')

def get_members_path(dstore_dir, model_type, dstore_size, dimension, sample_size, num_clusters, dstore_damb):
    if dstore_damb is None:
        return f'{dstore_dir}/members_{model_type}_{dstore_size}_{dimension}_{sample_size}_{num_clusters}.pkl'
    else:
        return f'{dstore_dir}/members_{model_type}_{dstore_size}_{dimension}_{dstore_damb}_{sample_size}_{num_clusters}.pkl'