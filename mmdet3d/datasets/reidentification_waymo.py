import itertools
import numpy as np
from mmdet.datasets import DATASETS

from .utils import (get_or_create_waymo_dict, set_seeds)
from .reidentification_base import ReIDDatasetBase



@DATASETS.register_module()
class ReIDDatasetWaymoFP(ReIDDatasetBase):

    def __init__(self,train,*args,**kwargs):
        sp = 'train' if train else 'val'
        self.instance_token_to_id = get_or_create_waymo_dict(f'instance_token_to_id_{sp}.pkl',
                                                            filepath='data/lstk/sparse/waymo',
                                                            infos_filepath=f'data/lstk/sparse/waymo/waymo_infos_{sp}_autolab.pkl')
        # print(args,kwargs)
        super().__init__(*args, **kwargs)
        self.obj_tokens = list(self.sparse_loader.obj_id_to_nums.keys())
        self.collect_dataset_idx()
        self.maintain_api()


    def before_collect_dataset_idx_hook(self):
        pass

    def after_collect_dataset_idx_hook(self):
        pass

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
            
            neg_obj_tok, l2, density = self.get_random_other_even_train(taken_idx=pos_obj_idx,
                                                                        taken_cls=l1,
                                                                        distribution=self.sparse_loader.obj_infos[pos_obj_tok]['distribution'])
            
            if neg_obj_tok.startswith("FP"):
                d2 = np.random.randn(self.subsample_dense,3)
                id2 = -1
            else:
                d2 = self.complete_loader[neg_obj_tok]
                id2 = self.instance_token_to_id[neg_obj_tok]
            
            neg_choice = self.sparse_loader.get_random_frame_even(neg_obj_tok,1,density=density,replace=False)[0]
            s2 = self.sparse_loader[(neg_obj_tok,neg_choice,)]
            
            return self.return_item(s1,s2,d1,d2,l1,l2,id1,id2)
    


@DATASETS.register_module()
class ReIDDatasetWaymoFPVal(ReIDDatasetWaymoFP):

    def __init__(self,max_combinations,*args,**kwargs):
        self.max_combinations = max_combinations
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.val_index)

    def after_collect_dataset_idx_hook(self):
        val_positives = []
        for i,c in zip(self.idx,self.classes):
            tok = self.obj_tokens[i]
            nums = self.sparse_loader.obj_id_to_nums[tok]
            combs = list(itertools.combinations(nums, r=2))
            np.random.shuffle(combs)
            combs=combs[:self.max_combinations] # dont use all combinations
            val_positives.extend([dict(o1=x[0],o2=x[1],tok=tok,cls=c) for x in combs])
        self.val_positives = val_positives

        val_negatives = []
        for x in self.val_positives:
            other_token,cls2 = self.get_random_other(taken_idx=x['o1'],taken_cls=x['cls'],)
            other_choice = self.get_random_frame(other_token,1,replace=False)[0]
            val_negatives.append(dict(o1=x['o1'],o2=other_choice,tok1=x['tok'],tok2=other_token,cls1=x['cls'],cls2=cls2))
        self.val_negatives = val_negatives
        self.val_index = np.arange(0,2*len(val_positives))


    def __getitem__(self,idx):
        if idx < len(self.val_positives):
            sample = self.val_positives[idx]
            pos_obj_tok = sample['tok']
            s1 = self.sparse_loader[(pos_obj_tok,sample['o1'],)]
            s2 = self.sparse_loader[(pos_obj_tok,sample['o2'],)]

            l1 = sample['cls']

            d1 = self.complete_loader[pos_obj_tok]
            id1 = self.instance_token_to_id[pos_obj_tok]

            v1 = self.sparse_loader.obj_infos[pos_obj_tok]['visibility'].get(int(sample['o1']),-1)
            v2 = self.sparse_loader.obj_infos[pos_obj_tok]['visibility'].get(int(sample['o2']),-1)

            return self.return_item_size_vis(s1,s2,d1,d1,l1,l1,id1,id1,v1,v2)
        else:
            idx = idx - len(self.val_positives)
            sample = self.val_negatives[idx]
            s1 = self.sparse_loader[(sample['tok1'],sample['o1'],)]
            s2 = self.sparse_loader[(sample['tok2'],sample['o2'],)]

            l1 = sample['cls1']
            l2 = sample['cls2']

            d1 = self.complete_loader[sample['tok1']]

            if sample['tok2'].startswith("FP"):
                d2 = np.random.randn(self.subsample_dense,3)
                id2 = -1
            else:
                d2 = self.complete_loader[sample['tok2']]
                id2 = self.instance_token_to_id[sample['tok2']]

            id1 = self.instance_token_to_id[sample['tok1']]

            v1 = self.sparse_loader.obj_infos[sample['tok1']]['visibility'].get(int(sample['o1']),-1)
            v2 = self.sparse_loader.obj_infos[sample['tok2']]['visibility'].get(int(sample['o2']),-1)
            
            return self.return_item_size_vis(s1,s2,d1,d2,l1,l2,id1,id2,v1,v2)
        



@DATASETS.register_module()
class ReIDDatasetWaymoFPValEven(ReIDDatasetWaymoFP):

    def __init__(self,max_combinations,test_mode,*args,**kwargs):
        self.max_combinations = max_combinations
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.val_index)


    def __getitem__(self,idx):
        if idx < len(self.val_positives):
            sample = self.val_positives[idx]
            pos_obj_tok = sample['tok']
            s1 = self.sparse_loader[(pos_obj_tok,sample['o1'],)]
            s2 = self.sparse_loader[(pos_obj_tok,sample['o2'],)]

            l1 = sample['cls']

            d1 = self.complete_loader[pos_obj_tok]
            id1 = self.instance_token_to_id[pos_obj_tok]

            v1 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v2,:2] ** 2).sum())

            return self.return_item_size_dist(s1,s2,d1,d1,l1,l1,id1,id1,v1,v2)
        else:
            idx = idx - len(self.val_positives)
            sample = self.val_negatives[idx]
            s1 = self.sparse_loader[(sample['tok1'],sample['o1'],)]
            s2 = self.sparse_loader[(sample['tok2'],sample['o2'],)]

            l1 = sample['cls1']
            l2 = sample['cls2']

            d1 = self.complete_loader[sample['tok1']]

            if sample['tok2'].startswith("FP"):
                d2 = np.random.randn(self.subsample_dense,3)
                id2 = -1
            else:
                d2 = self.complete_loader[sample['tok2']]
                id2 = self.instance_token_to_id[sample['tok2']]

            id1 = self.instance_token_to_id[sample['tok1']]

            v1 = self.sparse_loader.obj_infos[sample['tok1']]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = np.sqrt((self.sparse_loader.obj_infos[sample['tok1']]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[sample['tok2']]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = np.sqrt((self.sparse_loader.obj_infos[sample['tok2']]['all_sizes'][v2,:2] ** 2).sum())
            
            return self.return_item_size_dist(s1,s2,d1,d2,l1,l2,id1,id2,v1,v2)

    def before_collect_dataset_idx_hook(self):
        set_seeds(seed=self.validation_seed)
        
    def after_collect_dataset_idx_hook(self):
        val_positives = []
        for i,c in zip(self.idx,self.classes):
            tok = self.obj_tokens[i]
            nums = self.sparse_loader.obj_id_to_nums[tok]
            combs = list(itertools.combinations(nums, r=2))
            np.random.shuffle(combs)
            combs=combs[:self.max_combinations] # dont use all combinations
            val_positives.extend([dict(o1=x[0],
                                       o2=x[1],
                                       pts1=self.sparse_loader.obj_infos[tok]['num_pts'][x[0]],
                                       pts2=self.sparse_loader.obj_infos[tok]['num_pts'][x[1]],
                                       tok=tok,
                                       cls=c) 
                                    for x in combs])
        self.val_positives = val_positives


        self.sparse_loader.get_buckets(self.idx.tolist()+self.false_positive_idx.tolist())
        self.fp_buckets = self.sparse_loader.get_all_buckets(self.false_positive_idx.tolist())
        self.tp_buckets = self.sparse_loader.get_all_buckets(self.idx.tolist())

        val_negatives = []
        for x in self.val_positives:
            other_token, cls2, other_choice = self.get_random_other_even_val(taken_idx=x['o1'],
                                                                         taken_cls=x['cls'],
                                                                         pts=x['pts2'])

            # other_choice = self.get_random_frame_even(other_token,1,replace=False)[0]
            val_negatives.append(dict(o1=x['o1'],
                                      o2=other_choice,
                                      tok1=x['tok'],
                                      tok2=other_token,
                                      cls1=x['cls'],
                                      cls2=cls2))

        self.val_negatives = val_negatives
        self.val_index = np.arange(0,2*len(val_positives))


    

@DATASETS.register_module()
class ReIDDatasetWaymoImageFP(ReIDDatasetWaymoFP):

    def __init__(self,*args,vis_to_cls_id={1:0,2:1,3:2,4:3},**kwargs):
        super().__init__(*args, **kwargs)
        self.vis_to_cls_id = vis_to_cls_id

    def __getitem__(self,idx):
        # idx --> index into self.idx
        # obj_idx --> corresponding index into valid obj_tokens
        
        pos_obj_idx = self.idx[idx]
        l1 = self.classes[idx]
        
        pos_obj_tok = self.obj_tokens[pos_obj_idx]
        # d1 = self.complete_loader[pos_obj_tok]
        id1 = self.instance_token_to_id[pos_obj_tok]
        
        if np.random.choice([0,1]) == 1:
            pos_choices = self.get_random_frame(pos_obj_tok,2,replace=False)
            s1 = self.sparse_loader[(pos_obj_tok,pos_choices[0],)]
            s2 = self.sparse_loader[(pos_obj_tok,pos_choices[1],)]

            v1 = 1#self.sparse_loader.obj_infos[pos_obj_tok]['visibility'].get(int(pos_choices[0]),-1)
            v2 = 1#self.sparse_loader.obj_infos[pos_obj_tok]['visibility'].get(int(pos_choices[0]),-1)
                
            return self.return_item_im(s1,s2,l1,l1,v1,v2,id1,id1)
        else:
            pos_choice = self.get_random_frame(pos_obj_tok,1,replace=False)[0]
            s1 = self.sparse_loader[(pos_obj_tok,pos_choice,)]
            
            neg_obj_tok, l2, density = self.get_random_other_even_train(taken_idx=pos_obj_idx,
                                                                        taken_cls=l1,
                                                                        distribution=self.sparse_loader.obj_infos[pos_obj_tok]['distribution'])
            
            if neg_obj_tok.startswith("FP"):
                # d2 = np.random.randn(self.subsample_dense,3)
                id2 = -1
            else:
                # d2 = self.complete_loader[neg_obj_tok]
                id2 = self.instance_token_to_id[neg_obj_tok]
            
            neg_choice = self.sparse_loader.get_random_frame_even(neg_obj_tok,1,density=density,replace=False)[0]
            s2 = self.sparse_loader[(neg_obj_tok,neg_choice,)]



            v1 = 1#self.sparse_loader.obj_infos[pos_obj_tok]['visibility'].get(int(pos_choice),-1)
            v2 = 1#self.sparse_loader.obj_infos[neg_obj_tok]['visibility'].get(int(neg_choice),-1)
            
            
            return self.return_item_im(s1,s2,l1,l2,v1,v2,id1,id2)


@DATASETS.register_module()
class ReIDDatasetWaymoImageFPValEven(ReIDDatasetWaymoFPValEven,ReIDDatasetWaymoImageFP):

    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self,idx):
        if idx < len(self.val_positives):
            sample = self.val_positives[idx]
            pos_obj_tok = sample['tok']
            s1 = self.sparse_loader[(pos_obj_tok,sample['o1'],)]
            s2 = self.sparse_loader[(pos_obj_tok,sample['o2'],)]
            

            # print( self.sparse_loader.obj_infos[pos_obj_tok].keys())
            v1 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[pos_obj_tok]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = np.sqrt((self.sparse_loader.obj_infos[pos_obj_tok]['all_sizes'][v2,:2] ** 2).sum())

            l1 = sample['cls']

            id1 = self.instance_token_to_id[pos_obj_tok]

            sz1 = self.sparse_loader.obj_infos[pos_obj_tok]['num_pts'].get(int(sample['o1']),-1)
            sz2 = self.sparse_loader.obj_infos[pos_obj_tok]['num_pts'].get(int(sample['o2']),-1)

            return self.return_item_size_dist_im(s1,s2,l1,l1,v1,v2,id1,id1,sz1,sz2)
        else:
            idx = idx - len(self.val_positives)
            sample = self.val_negatives[idx]
            s1 = self.sparse_loader[(sample['tok1'],sample['o1'],)]
            s2 = self.sparse_loader[(sample['tok2'],sample['o2'],)]

            l1 = sample['cls1']
            l2 = sample['cls2']

            # print(self.sparse_loader.obj_infos[sample['tok1']].keys())
            v1 = self.sparse_loader.obj_infos[sample['tok1']]['nums_to_distance'].get(int(sample['o1']),-1)
            v1 = np.sqrt((self.sparse_loader.obj_infos[sample['tok1']]['all_sizes'][v1,:2] ** 2).sum())
            v2 = self.sparse_loader.obj_infos[sample['tok2']]['nums_to_distance'].get(int(sample['o2']),-1)
            v2 = np.sqrt((self.sparse_loader.obj_infos[sample['tok2']]['all_sizes'][v2,:2] ** 2).sum())

            if sample['tok2'].startswith("FP"):
                id2 = -1
            else:
                id2 = self.instance_token_to_id[sample['tok2']]

            id1 = self.instance_token_to_id[sample['tok1']]

            sz1 = self.sparse_loader.obj_infos[sample['tok1']]['num_pts'].get(int(sample['o1']),-1)
            sz2 = self.sparse_loader.obj_infos[sample['tok2']]['num_pts'].get(int(sample['o2']),-1)
            
            return self.return_item_size_dist_im(s1,s2,l1,l2,v1,v2,id1,id2,sz1,sz2)