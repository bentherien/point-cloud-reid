import json
import itertools
import torch
import time
import numpy as np
import torch.nn as nn

from mmdet.datasets import DATASETS

from functools import reduce

from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC

from mmdet3d.datasets.utils import get_or_create_nuscenes_dict, MatchingEval

from .utils import (set_seeds, make_tup_str, to_tensor, \
                    subsamplePC )

from .builder import build_dataset



@DATASETS.register_module()
class ReIDDatasetBase(object):
    
    def __init__(self,
                 CLASSES,
                 cls_to_idx,
                 cls_to_idx_fp,
                 tracking_classes,
                 tracking_classes_fp,
                 subsample_sparse,
                 subsample_dense,
                 return_mode='dict',
                 verbose=False,
                 validation_seed=0,
                 sparse_loader=dict(),
                 complete_loader=dict()):
        super().__init__()

        if verbose:
            print('Entering __init__() for ReIDDatasetBase')

        self.verbose = verbose
        self.return_mode = return_mode
        self.cls_to_idx = cls_to_idx
        self.idx_to_cls = {v:k for k,v in self.cls_to_idx.items()}
        self.cls_to_idx_fp = cls_to_idx_fp
        self.idx_to_cls_fp = {v:k for k,v in self.cls_to_idx_fp.items()}
        self.CLASSES = CLASSES
        self.tracking_classes = tracking_classes
        self.tracking_classes_fp = tracking_classes_fp
        self.validation_seed = validation_seed
        self.matching_eval = MatchingEval()
        self.subsample_sparse = subsample_sparse
        self.subsample_dense = subsample_dense


        # print(complete_loader)
        self.complete_loader = build_dataset(complete_loader)
        # print(sparse_loader)
        self.sparse_loader = build_dataset(sparse_loader)

    def maintain_api(self):
        # Maintain MMDetection3D API
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def eval_match(self, preds, targets, match_classes):
        log_vars = {}
        for k,v in self.cls_to_idx.items():
            idx = torch.where(match_classes[:,0] == v)
            if len(idx[0]) > 0:
                match_acc = (nn.Sigmoid()(preds[idx]) > 0.5).float().eq(targets[idx]).float().mean().item()
                log_vars['val_match_acc_{}'.format(k)] = match_acc

        idx = torch.where(match_classes.max(1).values >=len(self.CLASSES)) #false positives
        if len(idx[0]) > 0:
            match_acc = (nn.Sigmoid()(preds[idx]) > 0.5).float().eq(targets[idx]).float().mean().item()
            log_vars['val_match_acc_FP'] = match_acc

        log_vars.update(self.matching_eval.f1_precision_recall((nn.Sigmoid()(preds) > 0.5).float(), targets))

        return log_vars


    def evaluate(self,results,logger,neptune,**kwargs):
        t1 = time.time()
        rank, world_size = get_dist_info()
        device_id = torch.cuda.current_device()

        results_accum = {}
        for d in results:
            for k,v in d.items():
                if v != None:
                    results_accum[k] = results_accum.get(k,[]) + [v.to(device_id)]


        results_to_save = {}
        results = {k:torch.cat(v,dim=0) for k,v in results_accum.items()}
        to_delete = ['val_cls_gt','val_fp_gt','val_match_gt','val_vis_gt_all','val_vis_gt']
        
        if 'val_match_preds' in results:
            match_acc = (nn.Sigmoid()(results['val_match_preds']) > 0.5).float().eq(results['val_match_gt']).float().mean().item()
            to_log = self.eval_match(preds=results['val_match_preds'],targets=results['val_match_gt'],match_classes=results['match_classes'])
        
            results_to_save['results_per_distance'] = self.matching_eval.evaluate_distance(
                                                                preds=results['val_match_preds'],
                                                                targets=results['val_match_gt'], 
                                                                num_points=results['val_vis_gt_all'])

            results_to_save['results_per_visibility'] = self.matching_eval.eval_per_visibility(
                                                                preds=results['val_match_preds'],
                                                                targets=results['val_match_gt'], 
                                                                vis_classes=results['val_vis_gt_all'])

            results_to_save['results_per_points'] = self.matching_eval.evaluate_points(
                                                                preds=results['val_match_preds'],
                                                                targets=results['val_match_gt'],
                                                                num_points=results['num_points'])


            for cls_,idx in self.cls_to_idx.items():
                if idx == -1:
                    continue

                cls_filter = torch.where(torch.logical_or(results['match_classes'][:,0] == idx,
                                                          results['match_classes'][:,1] == idx))[0]

                if len(cls_filter) == 0:
                    continue

                results_to_save[cls_] = dict()
                results_to_save[cls_]['results_per_points'] = \
                    self.matching_eval.evaluate_points(preds=results['val_match_preds'][cls_filter],
                                                       targets=results['val_match_gt'][cls_filter],
                                                       num_points=results['num_points'][cls_filter,:])
                                                                            
                                                                            
            results_to_save = make_tup_str(results_to_save)
            json.dump(results_to_save,open('/tmp/results_per_visibility.json','w'))
            neptune['results_per_visibility'].upload('/tmp/results_per_visibility.json')

            to_delete += ['val_match_preds','match_classes','num_points','is_fp',]
        else:
            match_acc = None
            to_log = None

        if 'val_fp_preds' in results:
            fp_acc = (nn.Sigmoid()(results['val_fp_preds']) > 0.5).float().eq(results['val_fp_gt']).float().mean().item()
            to_delete += ['val_fp_preds']
        else:
            fp_acc = None

        if 'val_cls_preds' in results:
            cls_acc = results['val_cls_preds'].argmax(dim=1).eq(results['val_cls_gt']).float().mean().item()
            to_delete += ['val_cls_preds']
        else:
            cls_acc = None

        if 'val_vis_preds' in results:
            vis_acc = results['val_vis_preds'].argmax(dim=1).eq(results['val_vis_gt']).float().mean().item()
            to_delete += ['val_vis_preds']
        else:
            vis_acc = None


        for x in to_delete:
            try:
                del results[x]
            except KeyError:
                print("KeyError in evaluate() no key {}".format(x))


        results = {k:v.mean().item() for k,v in results.items()}

        if match_acc:
            results['val_match_acc'] = match_acc

        if cls_acc:
            results['val_cls_acc'] = cls_acc
            
        if vis_acc:
            results['val_vis_acc'] = vis_acc

        if fp_acc:
            results['val_fp_acc'] = fp_acc

        if to_log:
            results.update(to_log)

        print("Evaluation took {} seconds".format(time.time() - t1))
        for k,v in results.items():
            print(k,round(v,6))

        json.dump(results,open('/tmp/overall_results.json','w'))
        neptune['overall_results'].upload('/tmp/overall_results.json')

        return results

    
    def collect_dataset_idx(self,):
        self.before_collect_dataset_idx_hook()

        idx = np.arange(0,len(self.sparse_loader.obj_id_to_nums))
        
        #keep only those with at least two positive examples
        temp = np.array([len(v) for k,v in self.sparse_loader.obj_id_to_nums.items()])
        is_fp = np.array([k.startswith('FP') for k in self.sparse_loader.obj_id_to_nums.keys()])
        

        ##########################################
        #get true positive indices
        ##########################################
        keep_idx = np.where(reduce(np.logical_and,[temp > 2, is_fp == 0]))
        self.idx = idx[keep_idx]
        self.classes = np.array([self.cls_to_idx[self.tracking_classes.get(
                                            self.sparse_loader.obj_infos[self.obj_tokens[x]]['class_name'], 
                                            'none_key',)]
                                       for x in self.idx])
        #only keep tracked classes
        keep_idx = np.where(self.classes != -1)
        self.idx = self.idx[keep_idx]
        self.classes = self.classes[keep_idx]

        ##########################################
        #get false positive indices
        ##########################################
        keep_fp = np.where(reduce(np.logical_and,[temp > 0, is_fp == 1])) #keep all false positives with at least on example
        self.false_positive_idx = idx[keep_fp]
        self.false_positive_classes = np.array([self.cls_to_idx[self.tracking_classes_fp.get(self.sparse_loader.obj_infos[self.obj_tokens[x]]['class_name'], 
                                                                                          'none_key',)]
                                                for x in self.false_positive_idx])

        #only keep tracked classes
        keep_idx = np.where(self.false_positive_classes != -1)
        self.false_positive_idx = self.false_positive_idx[keep_idx]
        self.false_positive_classes = self.false_positive_classes[keep_idx]
        self.false_positive_classes += len(self.CLASSES) #offset by 10 to avoid overlap with true classes
        

        self.shuffle_idx()
        
        if len(self.false_positive_idx) > 0:
            assert np.min(temp[self.false_positive_idx]) >= 1
        
        assert np.min(temp[self.idx]) >= 2, "not enough positive examples {}".format(self.idx)

        self.after_collect_dataset_idx_hook()
    
    def before_collect_dataset_idx_hook(self):
        raise NotImplementedError("this function should be implemented by the child class")
    
    def after_collect_dataset_idx_hook(self):
        raise NotImplementedError("this function should be implemented by the child class")

    def get_random_frame_even(self,*args,**kwargs):
        return self.sparse_loader.get_random_frame_even(*args,**kwargs)
        
    def shuffle_idx(self):
        p = np.random.permutation(len(self.idx))
        self.idx = self.idx[p]
        self.classes = self.classes[p]
    
    def get_random_frame(self,*args,**kwargs):
        return self.sparse_loader.get_random_frame(*args,**kwargs)
    
    def get_random_other(self,taken_idx,taken_cls):
        """get a random index other than the one specified while ensuring 
            the corresponding class is the same"""
        temp_idx = self.idx[np.where(self.classes == taken_cls)[0]]

        if len(temp_idx) == 1:
            raise AttributeError(f"self.idx: {temp_idx} is of invalid size 1. This will cause an infinite loop.")
            
        other = taken_idx
        while(other == taken_idx):
            other = np.random.choice(temp_idx,1)[0]
        
        return self.obj_tokens[other], taken_cls

    def get_random_other_fp(self,taken_idx,taken_cls):
        """get a random index other than the one specified including false positives
            while ensuring the corresponding class is the same"""

        if np.random.choice([0,1]) == 1:
            # sample True Positive

            temp_idx = self.idx[np.where(self.classes == taken_cls)[0]]

            if len(temp_idx) == 1:
                raise AttributeError(f"self.idx: {temp_idx} is of invalid size 1. This will cause an infinite loop.")
                
            other = taken_idx
            while(other == taken_idx):
                other = np.random.choice(temp_idx,1)[0]
        else:
            taken_cls += len(self.CLASSES)

            # Sample False Positive
            temp_idx = self.false_positive_idx[np.where(self.false_positive_classes == taken_cls)[0]]


            if len(temp_idx) == 0:
                print('self.false_positive_idx',self.false_positive_idx)
                print('self.false_positive_classes',self.false_positive_classes)
                print('cls_idx',taken_cls)
                print('temp_idx',temp_idx)
                raise AttributeError(f"self.idx: {temp_idx} is of invalid size 0. This will cause an infinite loop.")
                
            other = taken_idx
            while(other == taken_idx):
                other = np.random.choice(temp_idx,1)[0]


        return self.obj_tokens[other], taken_cls


    def get_random_other_even_train(self,taken_idx,taken_cls,distribution):
        """get a random index other than the one specified while ensuring 
            the corresponding class is the same"""

        density = np.random.choice(np.arange(len(self.sparse_loader.buckets)),p=distribution)
        b = self.sparse_loader.buckets[density]

        if np.random.choice([0,1]) == 1:
            # sample True Positive
            class_name = self.idx_to_cls_fp[taken_cls]
        else:
            #sample False Positive 
            taken_cls += len(self.CLASSES)
            class_name = self.idx_to_cls_fp[taken_cls]


        tok_list, density = self.sparse_loader.get_class_list_density(class_name=class_name,density_idx=density)

        if len(tok_list) <= 1:
            print('class_name', class_name)
            print('tok_list', tok_list)
            print('taken_idx', taken_idx)
            print('taken_cls', taken_cls)
            print('density', density)
            raise AttributeError(f"fps: {tok_list} is of invalid size 0. This will cause an infinite loop.")
            
        other_token = self.obj_tokens[taken_idx]
        count = 0
        while(other_token == self.obj_tokens[taken_idx]):
            other = np.random.choice(len(tok_list),1)[0]
            other_token = tok_list[other][0] # get token from tuple
            count += 1
            if count > 100000:
                print('tok_list', tok_list)
                print('taken_idx', taken_idx)
                print('taken_cls', taken_cls)
                print('density', density)
                raise AttributeError("Infinite Loop detected. This is likely due to a bug in the code.")


        return other_token, taken_cls, density
    
    def get_random_other_even_val(self,taken_idx,taken_cls, pts):
        """get a random index other than the one specified while ensuring 
           the corresponding class is the same and the number of points fall within the same bucket"""

        # Get the bucket
        b_idx = int(self.sparse_loader.special_log(pts))
        b = self.sparse_loader.buckets[b_idx]

        if np.random.choice([0,1]) == 1:
            # sample True Positive

            #get possible samples
            extracted = False
            while not extracted:
                try:
                    tps = self.tp_buckets[self.idx_to_cls_fp[taken_cls]][b]
                    if len(tps) == 1:
                        b_idx -= 1
                        b = self.sparse_loader.buckets[b_idx]
                        # print('lowering idx for TP bucket',b)
                        continue

                    extracted = True
                except:
                    b_idx -= 1
                    b = self.sparse_loader.buckets[b_idx]
                    # print('lowering idx for TP bucket',b)

            if len(tps) == 1:
                raise AttributeError(f"tps[other]: {tps} is of invalid size 1. This will cause an infinite loop.")
                
            other_token = self.obj_tokens[taken_idx]
            while(other_token == self.obj_tokens[taken_idx]):
                other = np.random.choice(len(tps),1)[0]
                other_token = tps[other][0] # get token from tuple
        else:
            # Sample False Positive

            #increment class
            taken_cls += len(self.CLASSES)
            extracted = False
            while not extracted:
                try:
                    fps = self.fp_buckets[self.idx_to_cls_fp[taken_cls]][b]
                    extracted = True
                except KeyError:
                    # print("For class :{}".format(self.idx_to_cls_fp[taken_cls]),
                        #   sorted([(k,len(v),) for k,v in self.fp_buckets.items()],key=lambda x: x[1]))
                    b_idx -= 1
                    b = self.sparse_loader.buckets[b_idx]
                    # print('lowering idx for FP bucket',b)

            if len(fps) == 0:
                print('b',b)
                print('cls_idx',taken_cls)
                print('fps',fps)
                raise AttributeError(f"fps: {fps} is of invalid size 0. This will cause an infinite loop.")
                
            other_token = self.obj_tokens[taken_idx]
            while(other_token == self.obj_tokens[taken_idx]):
                other = np.random.choice(len(fps),1)[0]
                other_token = fps[other][0] # get token from tuple
        
        frame = np.random.choice(self.sparse_loader.obj_infos[other_token]['buckets'][b],1)[0]
        return other_token, taken_cls, frame


    def return_item(self,s1,s2,d1,d2,l1,l2,id1,id2):
        s1,s2 = subsamplePC(np.moveaxis(s1,0,1),self.subsample_sparse), subsamplePC(np.moveaxis(s2,0,1),self.subsample_sparse)
        d1,d2 = subsamplePC(np.moveaxis(d1,0,1),self.subsample_dense), subsamplePC(np.moveaxis(d2,0,1),self.subsample_dense)

        s1,s2,d1,d2,l1,l2,id1,id2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(d1)),DC(to_tensor(d2)),\
                                    DC(to_tensor(l1)),DC(to_tensor(l2)),DC(to_tensor(id1)),DC(to_tensor(id2))
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, dense_1=d1, dense_2=d2, label_1=l1, label_2=l2, id_1=id1, id_2=id2)
        elif self.return_mode == 'tuple':
            return s1,s2,d1,d2,l1,l2,id1,id2
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")
    
    def return_item_size(self,s1,s2,d1,d2,l1,l2,id1,id2):
        sz1,sz2 = DC(to_tensor(s1.shape[0])),DC(to_tensor(s2.shape[0]))
        s1,s2 = subsamplePC(np.moveaxis(s1,0,1),self.subsample_sparse), subsamplePC(np.moveaxis(s2,0,1),self.subsample_sparse)
        d1,d2 = subsamplePC(np.moveaxis(d1,0,1),self.subsample_dense), subsamplePC(np.moveaxis(d2,0,1),self.subsample_dense)

        s1,s2,d1,d2,l1,l2,id1,id2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(d1)),DC(to_tensor(d2)),\
                                    DC(to_tensor(l1)),DC(to_tensor(l2)),DC(to_tensor(id1)),DC(to_tensor(id2))
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, dense_1=d1, dense_2=d2, label_1=l1, label_2=l2, id_1=id1, id_2=id2, size_1=sz1, size_2=sz2)
        elif self.return_mode == 'tuple':
            return s1,s2,d1,d2,l1,l2,id1,id2,sz1,sz2
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")


    def return_item_size_vis(self,s1,s2,d1,d2,l1,l2,id1,id2,v1,v2):

        if v1 == None:
            v1 = -1 

        if v2 == None:
            v2 = -1

        if type(v1) == str:
            v1 = int(v1)
            v2 = int(v2)

        v1 = self.vis_to_cls_id.get(v1,-1)
        v2 = self.vis_to_cls_id.get(v2,-1)

        vis1,vis2 = DC(to_tensor(v2)),DC(to_tensor(v1))
        sz1,sz2 = DC(to_tensor(s1.shape[0])),DC(to_tensor(s2.shape[0]))
        s1,s2 = subsamplePC(np.moveaxis(s1,0,1),self.subsample_sparse), subsamplePC(np.moveaxis(s2,0,1),self.subsample_sparse)
        d1,d2 = subsamplePC(np.moveaxis(d1,0,1),self.subsample_dense), subsamplePC(np.moveaxis(d2,0,1),self.subsample_dense)

        s1,s2,d1,d2,l1,l2,id1,id2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(d1)),DC(to_tensor(d2)),\
                                    DC(to_tensor(l1)),DC(to_tensor(l2)),DC(to_tensor(id1)),DC(to_tensor(id2))
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, dense_1=d1, dense_2=d2, label_1=l1, label_2=l2, 
                        id_1=id1, id_2=id2, size_1=sz1, size_2=sz2,vis_1=vis1,vis_2=vis2)
        elif self.return_mode == 'tuple':
            return s1,s2,d1,d2,l1,l2,id1,id2,sz1,sz2,vis1,vis2
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")


    def return_item_size_dist(self,s1,s2,d1,d2,l1,l2,id1,id2,v1,v2):
        if v1 == None:
            v1 = -1.
        if v2 == None:
            v2 = -1.

        vis1,vis2 = DC(to_tensor(v2)),DC(to_tensor(v1))
        sz1,sz2 = DC(to_tensor(s1.shape[0])),DC(to_tensor(s2.shape[0]))
        s1,s2 = subsamplePC(np.moveaxis(s1,0,1),self.subsample_sparse), subsamplePC(np.moveaxis(s2,0,1),self.subsample_sparse)
        d1,d2 = subsamplePC(np.moveaxis(d1,0,1),self.subsample_dense), subsamplePC(np.moveaxis(d2,0,1),self.subsample_dense)

        s1,s2,d1,d2,l1,l2,id1,id2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(d1)),DC(to_tensor(d2)),\
                                    DC(to_tensor(l1)),DC(to_tensor(l2)),DC(to_tensor(id1)),DC(to_tensor(id2))
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, dense_1=d1, dense_2=d2, label_1=l1, label_2=l2, 
                        id_1=id1, id_2=id2, size_1=sz1, size_2=sz2,vis_1=vis1,vis_2=vis2)
        elif self.return_mode == 'tuple':
            return s1,s2,d1,d2,l1,l2,id1,id2,sz1,sz2,vis1,vis2
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")

    def return_item_im(self,s1,s2,l1,l2,v1,v2,id1,id2):
        if v1 == None:
            v1 = -1 

        if v2 == None:
            v2 = -1

        if type(v1) == str:
            v1 = int(v1)
            v2 = int(v2)

        v1 = self.vis_to_cls_id.get(v1,-1)
        v2 = self.vis_to_cls_id.get(v2,-1)
            
        s1,s2,l1,l2,v1,v2,id1,id2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(l1)),\
                                    DC(to_tensor(l2)),DC(to_tensor(v1)),DC(to_tensor(v2)),\
                                    DC(to_tensor(id1)),DC(to_tensor(id2))
        
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, label_1=l1, 
                        label_2=l2, vis_1=v1, vis_2=v2, 
                        id_1=id1, id_2=id2)
        
        elif self.return_mode == 'tuple':
            return s1,s2,l1,l2,v1,v2,id1,id2
        
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")
        



    def return_item_size_dist_im(self,s1,s2,l1,l2,v1,v2,id1,id2,sz1,sz2):
        if v1 == None:
            v1 = -1.
        if v2 == None:
            v2 = -1.
            
        s1,s2,l1,l2,v1,v2,id1,id2,sz1,sz2 = DC(to_tensor(s1)),DC(to_tensor(s2)),DC(to_tensor(l1)),\
                                            DC(to_tensor(l2)),DC(to_tensor(v1)),DC(to_tensor(v2)),\
                                            DC(to_tensor(id1)),DC(to_tensor(id2)),DC(to_tensor(sz2)),DC(to_tensor(sz1))
        
        if self.return_mode == 'dict':
            return dict(sparse_1=s1, sparse_2=s2, label_1=l1, 
                        label_2=l2, vis_1=v1, vis_2=v2, 
                        id_1=id1, id_2=id2, size_1=sz1,size_2=sz2)
        
        elif self.return_mode == 'tuple':
            return s1,s2,l1,l2,v1,v2,id1,id2,sz1,sz2
        
        else:
            raise AttributeError(f"Invalid return mode {self.return_mode}")

        
        
    def __len__(self):
        return len(self.idx)
        
    def __getitem__(self,idx):
        # idx --> index into self.idx
        # obj_idx --> corresponding index into valid obj_tokens
        
        pos_obj_idx = self.idx[idx]
        l1 = self.classes[idx]
        
        pos_obj_tok = self.obj_tokens[pos_obj_idx]
        d1 = self.complete_loader[pos_obj_tok]
        id1 = self.instance_token_to_id[pos_obj_tok]
        
        if np.random.choice([0,1]) == 1:
            pos_choices = self.get_random_frame(pos_obj_tok,2,replace=False)
            s1 = self.sparse_loader[(pos_obj_tok,pos_choices[0],)]
            s2 = self.sparse_loader[(pos_obj_tok,pos_choices[1],)]

            return self.return_item(s1,s2,d1,d1,l1,l1,id1,id1)
        else:
            pos_choice = self.get_random_frame(pos_obj_tok,1,replace=False)[0]
            s1 = self.sparse_loader[(pos_obj_tok,pos_choice,)]
            
            neg_obj_tok, l2 = self.get_random_other(taken_idx=pos_obj_idx,taken_cls=l1)
            
            if neg_obj_tok.startswith("FP"):
                d2 = np.random.randn(self.subsample_dense,3)
                id2 = -1
            else:
                d2 = self.complete_loader[neg_obj_tok]
                id2 = self.instance_token_to_id[neg_obj_tok]
            
            neg_choice = self.get_random_frame(neg_obj_tok,1,replace=False)[0]
            s2 = self.sparse_loader[(neg_obj_tok,neg_choice,)]
            
            return self.return_item(s1,s2,d1,d2,l1,l2,id1,id2)

