import os
import time

import os.path as osp
import numpy as np
import pickle as pkl

from mmdet.datasets import DATASETS
from lamtk.aggregation.loader import Loader
from lamtk.aggregation.utils import filter_metadata_by_scene_ids, combine_metadata
from mmdet3d.datasets.utils import get_or_create_nuscenes_dict

from pathlib import Path


def combine_metadata_fix(metadata_path_sparse):
    """Combines metadata from multiple files into one dictionary, and fixes the 
    path to the data for files store under new_datasets."""

    md_files = os.listdir(metadata_path_sparse)
    sparse_metadata = {x:pkl.load(open(os.path.join(metadata_path_sparse,x),'rb')) \
                                            for x in md_files}

    # for x in md_files:
    #     temp = [int(y) for y in x[-11:-4].split("-")]
    #     if 500 <= temp[0] and temp[1] <= 700:
    #         print(x)
    #         for k in sparse_metadata[x]['obj_infos']:
    #             sparse_metadata[x]['obj_infos'][k]['path'] = Path("../../../new_datasets/lstk/sparse-waymo-det-both-train") \
    #                                                          / sparse_metadata[x]['obj_infos'][k]['path']


    metadata = dict(scene_infos={}, obj_infos={}, frame_infos={})
    for metadata_i in sparse_metadata.values():

        metadata['scene_infos'].update(metadata_i['scene_infos'])
        metadata['obj_infos'].update(metadata_i['obj_infos'])
        metadata['frame_infos'].update(metadata_i['frame_infos'])
    return metadata


def load_metadata(metadata_path,use_fix=False):
    if metadata_path.endswith('.pkl'):
            with open(metadata_path, 'rb') as f:
                sparse_metadata = pkl.load(f)
    elif use_fix == True:
        sparse_metadata = combine_metadata_fix(metadata_path)
    else:
         sparse_metadata = combine_metadata([pkl.load(open(os.path.join(metadata_path,x),'rb')) \
                                                for x in os.listdir(metadata_path)])
         
    return sparse_metadata


def load_nuscenes_metadata_split(metadata_path,version,train):
    """Loads a specific split of the nuScenes dataset.
    
    Args:
        metadata_path (str): Path to the metadata file.
        version (str): Version of the dataset to load.
        train (bool): Whether to load the training or validation split.
        use_metadata_fix (bool): Whether to use the metadata fix(only relevant to ).
    """

    splits_by_scene = get_or_create_nuscenes_dict(filename='ds_name_to_scene_token.pkl',
                                                        filepath='data/lstk',
                                                        nuscenes_dataroot='data/nuscenes')

    metadata = load_metadata(metadata_path,use_fix=False)
    split = splits_by_scene[version]['train'].values() if train else splits_by_scene[version]['val'].values()

    return filter_metadata_by_scene_ids(metadata,split)


@DATASETS.register_module()
class ObjectLoaderSparseBase(Loader):

    def __init__(self, tracking_classes, min_points, use_distance, filter_mode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filter_mode = filter_mode
        self.min_points = min_points
        self.use_distance = use_distance
        self.tracking_classes = tracking_classes
        if self.filter_mode not in ['both','vis','pts','pts and vis']:
            raise ValueError("filter_mode must be one of ['both','visibility','pts','pts and vis']")

    def load(self, *args, **kwargs):
        raise NotImplementedError("This method shouw be implemented in the child class")
    
    def __getitem__(self, x):
        obj_id, frame_id = x
        obj = self.obj_infos.get(obj_id, None)
        if obj:
            return self.load(obj,str(frame_id))
        else:
            raise ValueError(f'obj_id {obj_id} does not exist in obj_infos')
        

    def get_filtered_nums(self,obj_key,obj_entry,min_points,only_keyframes=False):

        if self.filter_mode == 'both':
            vis_num_keys = list(obj_entry['visibility'].keys())
            pts_num_keys = list(obj_entry['num_pts'].keys())
            nums = vis_num_keys + pts_num_keys
            nums = list(set(nums))
            nums = sorted(nums,key=lambda x : int(x))
            if len(vis_num_keys) != len(pts_num_keys):
                print(len(vis_num_keys),len(pts_num_keys))

        elif self.filter_mode == 'vis':
           nums = list(obj_entry['visibility'].keys())
           nums = sorted(nums,key=lambda x : int(x))

        elif self.filter_mode == 'pts':
            nums = sorted(list(obj_entry['num_pts'].keys()),key=lambda x : int(x))
            temp = np.array([obj_entry['num_pts'][int(x)] for x in nums])
            obj_filter = np.where(temp >= min_points)

            t = np.where(temp == 0)
            # if len(t[0]) > 0:
            #     print("num zero samples:",len(t[0]))
            nums = np.array(nums)[obj_filter]

        elif self.filter_mode == 'pts and vis':
            vis_num_keys = set(list(obj_entry['visibility'].keys()))
            pts_num_keys = sorted(list(obj_entry['num_pts'].keys()),key=lambda x : int(x))
            temp = np.array([obj_entry['num_pts'][int(x)] for x in pts_num_keys])
            obj_filter = np.where(temp >= min_points)
            pts_num_keys = set(list(np.array(pts_num_keys)[obj_filter]))
            nums = list(vis_num_keys.intersection(pts_num_keys))
            nums = sorted(nums,key=lambda x : int(x))
        else:
            raise NotImplementedError("get_filtered_nums() in not implemented for self.filter_mode :{} ".format(self.filter_mode)) 

        if self.use_distance:
            nums_to_distance = dict()
            for i,num in enumerate(obj_entry['num_pts'].keys()):
                nums_to_distance[num] = i
            self.obj_infos[obj_key]['nums_to_distance'] = nums_to_distance

        return nums

    def collect_obj_id_to_nums(self,min_points,only_keyframes=False):
        obj_id_to_nums = {}
        for k,v in self.obj_infos.items():
            obj_id_to_nums[k] = self.get_filtered_nums(k,v,min_points,only_keyframes)
        return obj_id_to_nums
        
    def get_random_frame(self,obj_tok,num_samples,replace=False):
        nums = self.obj_id_to_nums[obj_tok]
        assert len(nums) >= num_samples, f"Assertion Error nums {nums} encountered for obj_tok:{obj_tok}"
        return np.random.choice(nums,num_samples,replace=replace)
        
    def get_buckets(self,index):
        """This function is used to calculated the distribution of the number of 
            points in each object."""
        self.buckets = [(2**x,2**(x+1)) for x in range(20)]
        temp = list(self.obj_id_to_nums.keys())
        for idx in index:
            obj = self.obj_infos[temp[idx]]
            obj_buckets = {}
            for n in self.obj_id_to_nums[obj['id']]:
                npts = obj['num_pts'][n]
                key = self.buckets[int(self.special_log(npts))]
                try:
                    obj_buckets[key] += [n]
                except KeyError:
                    obj_buckets[key] = [n]
            obj['buckets'] = obj_buckets
            obj['distribution'] = np.array([len(obj['buckets'].get(x,[])) for x in self.buckets])
            obj['distribution'] = obj['distribution']/obj['distribution'].sum()

    def get_all_buckets(self,index):
        """This function is used to accumulated the distribution of the power two 
        buckets over all the objects
        """
        all_buckets = {}
        temp = list(self.obj_id_to_nums.keys())
        for idx in index:
            obj = self.obj_infos[temp[idx]]
            cls_temp = self.tracking_classes.get(obj['class_name'],None)
            if cls_temp is None:
                #skip non tracking classes
                continue

            if obj['id'].startswith('FP_'):
                cls_ = 'FP_' + cls_temp
            else:
                cls_ = cls_temp

            all_buckets[cls_] = all_buckets.get(cls_,{})
            for k,pts_list in obj['buckets'].items():
                try:
                    all_buckets[cls_][k].append((obj['id'],len(pts_list),))
                except KeyError:
                    all_buckets[cls_][k] = [(obj['id'],len(pts_list),)]

        self.all_buckets = all_buckets
        return self.all_buckets

    def get_random_frame_even(self,obj_tok,num_samples,density,replace=False):
        """Finds and observation of a given object that has a certain density."""
        obj_buckets = self.obj_infos[obj_tok]['buckets']
        while(len(obj_buckets.get(self.buckets[density],[])) == 0):
            density -= 1
            if density == -1:
                density = 0 
                while(len(obj_buckets.get(self.buckets[density],[])) == 0):
                    density += 1
                    if density >= len(self.buckets):
                        print(obj_buckets)
                        print(self.obj_infos[obj_tok])
                        raise Exception("Infinite loop will occur in get_random_frame_even()")
        nums = obj_buckets.get(self.buckets[density],[])
        return np.random.choice(nums,num_samples,replace=replace)

    def get_class_list_density(self,class_name,density_idx):
        """Finds a density bucket that has at least 2 objects in it.
        
        The density bucket is first found by going down the list of buckets until a bucket 
        with at least 2 objects is found. If no such bucket is found, the function goes up the list 
        of buckets until a bucket with at least 2 objects is found. If no such bucket is found, 
        an exception is raised.
    
        Arguments:
            class_name {str} -- The class name of the object
            density_idx {int} -- The index of the density bucket to start searching from
        """
        while(len(self.all_buckets[class_name].get(self.buckets[density_idx],[])) <= 1):
            density_idx -= 1
            if density_idx == -1:
                density_idx = 0 
                while(len(self.all_buckets[class_name].get(self.buckets[density_idx],[])) <= 1):
                    density_idx += 1
                    if density_idx >= len(self.buckets):
                        raise Exception("Infinite loop will occur in get_random_frame_even()")
        
        return self.all_buckets[class_name][self.buckets[density_idx]],density_idx
    

    def special_log(self,n):
        if n == 0:
            return -1
        return np.log2(n)
    

    def load_points(self, info, frame_idx):
        points = []
        if 'pts_data' in info:
            for name in self.load_feats:
                points.append(info['pts_data'][f'pts_{name}'])
        elif 'path' in info:
            path = info['path']
            try:
                self.obj_id_to_nums[info['id']]
            except KeyError: 
                self.obj_id_to_nums[info['id']] = self.get_filtered_nums(info)
            path = osp.join(info['path'], frame_idx)
                            #self.obj_id_to_nums[info['id']][frame_idx])
            for name, dim in zip(self.load_feats, self.load_dims):
                feats_file = f'{self.data_root}/{path}/pts_{name}.bin'
                num_pts = int(os.stat(feats_file).st_size // (4 * dim))
                num_pts -= int(num_pts * self.load_fraction)
                points.append(np.fromfile(feats_file,
                                          offset=4 * dim * num_pts,
                                          dtype=np.float32).reshape(-1, dim))
        else:
            raise ValueError(f'info must have either path or pts_data')
        return np.concatenate(points, axis=-1)


    def load_image(self, info, frame_idx):
        if 'path' in info:
            path = info['path']
            try:
                self.obj_id_to_nums[info['id']]
            except KeyError: 
                self.obj_id_to_nums[info['id']] = self.get_filtered_nums(info)

            # print(os.listdir(self.data_root))
            path = osp.join(info['path'], frame_idx)
            feats_file = f'{self.data_root}/{path}/img_crop.bin'

            try:
                im = np.fromfile(feats_file,dtype=np.float32).reshape((-1,)+info['crop_size'])
                if tuple(im.shape[1:]) != self.crop_size:
                    im = im[:,0:self.crop_size[0],0:self.crop_size[1]]

            except FileNotFoundError:
                # print('[WARNING] FileNotFoundError in ObjectImageLoaderSparseWaymo No such file "{}" loading ZEROs image instead...'.format(
                #     feats_file))
                im = np.zeros((3,) + self.crop_size)
            except KeyError:
                print('[WARNING] KeyError in ObjectImageLoaderSparseWaymo for crop_size loading ZEROs image instead...')
                im = np.zeros((3,) + self.crop_size)

            return im #np.swapaxes(im.T,0,1)
            
    
        else:
            raise ValueError(f'info must have path')


############################################
# Nuscenes
############################################

    
@DATASETS.register_module()
class ObjectLoaderSparseNuscenes(ObjectLoaderSparseBase):
    def __init__(self,
                 train,
                 version,
                 metadata_path,
                 *args,
                 use_metdata_fix=False,
                 **kwargs):
        metadata = load_nuscenes_metadata_split(metadata_path,version,train)
        super().__init__(*args,metadata=metadata,**kwargs)

        self.sample_to_keyframes = get_or_create_nuscenes_dict(filename='sample_to_keyframes.pkl',
                                                                        filepath='data/lstk',
                                                                        nuscenes_dataroot='data/nuscenes')

        self.instance_to_keyframes = get_or_create_nuscenes_dict(filename='instance_to_keyframes.pkl',
                                                                        filepath='data/lstk',
                                                                        nuscenes_dataroot='data/nuscenes')

        self.obj_id_to_nums = self.collect_obj_id_to_nums(self.min_points)

        t1 = time.time()
        self.get_buckets(np.arange(0,len(self.obj_id_to_nums)))
        self.get_all_buckets(np.arange(0,len(self.obj_id_to_nums)))
        print("Loading buckets took: ",time.time()-t1)

    def load(self,*args,**kwerags):
        return self.load_points(*args,**kwerags)


@DATASETS.register_module()
class ObjectLoaderSparseNuscenesImage(ObjectLoaderSparseNuscenes):
    def __init__(self,*args,crop_size=(224,224),**kwargs):
        super().__init__(*args,**kwargs)
        self.crop_size = crop_size
        
    def load(self,*args,**kwargs):
        return self.load_image(*args,**kwargs)



############################################
# Waymo
############################################


@DATASETS.register_module()
class ObjectLoaderSparseWaymo(ObjectLoaderSparseBase):
    def __init__(self,
                 metadata_path,
                 *args,
                 use_metdata_fix=False,
                 **kwargs):

        self.zero_samples = 0
        metadata = load_metadata(metadata_path,use_fix=use_metdata_fix)
        super().__init__(*args,metadata=metadata,**kwargs)

        self.obj_id_to_nums = self.collect_obj_id_to_nums(self.min_points)

        t1 = time.time()
        self.get_buckets(np.arange(0,len(self.obj_id_to_nums)))
        self.get_all_buckets(np.arange(0,len(self.obj_id_to_nums)))
        print("Loading buckets took: ",time.time()-t1)
        print("zero samples: ",self.zero_samples)


    def get_filtered_nums(self,obj_key,obj_entry,min_points,only_keyframes=False):

        if self.filter_mode == 'both':
            vis_num_keys = list(obj_entry['box2d'].keys())
            pts_num_keys = list(obj_entry['num_pts'].keys())
            nums = vis_num_keys + pts_num_keys
            nums = list(set(nums))
            nums = sorted(nums,key=lambda x : int(x))
            if len(vis_num_keys) != len(pts_num_keys):
                print(len(vis_num_keys),len(pts_num_keys))

        elif self.filter_mode == 'vis':
           nums = list(obj_entry['box2d'].keys())
           nums = sorted(nums,key=lambda x : int(x))

        elif self.filter_mode == 'pts':
            nums = sorted(list(obj_entry['num_pts'].keys()),key=lambda x : int(x))
            temp = np.array([obj_entry['num_pts'][int(x)] for x in nums])
            obj_filter = np.where(temp >= min_points)
            nums = np.array(nums)[obj_filter]

            t = np.where(temp == 0)
            if len(t[0]) > 0:
                self.zero_samples += len(t[0])
                # print("num zero samples:",len(t[0]))

        elif self.filter_mode == 'pts and vis':
            if type(obj_entry['box2d']) == list:
                print(obj_entry['box2d'])
                return []
            else:
                # print("Not a list!!")
                pass
            #     if obj_entry['box2d'][0] != None:
            #         print(obj_entry['box2d'])
            #         exit(0)
            #     return []

            vis_num_keys = set(list(obj_entry['box2d'].keys()))
            pts_num_keys = sorted(list(obj_entry['num_pts'].keys()),key=lambda x : int(x))
            temp = np.array([obj_entry['num_pts'][int(x)] for x in pts_num_keys])
            obj_filter = np.where(temp >= min_points)
            pts_num_keys = set(list(np.array(pts_num_keys)[obj_filter]))
            nums = list(vis_num_keys.intersection(pts_num_keys))
            nums = sorted(nums,key=lambda x : int(x))
        else:
            raise NotImplementedError("get_filtered_nums() in not implemented for self.filter_mode :{} ".format(self.filter_mode)) 

        if self.use_distance:
            nums_to_distance = dict()
            for i,num in enumerate(obj_entry['num_pts'].keys()):
                nums_to_distance[num] = i
            self.obj_infos[obj_key]['nums_to_distance'] = nums_to_distance

        return nums

    def load(self,*args,**kwerags):
        return self.load_points(*args,**kwerags)


@DATASETS.register_module()
class ObjectLoaderSparseWaymoImage(ObjectLoaderSparseWaymo):
    def __init__(self,*args,crop_size=(224,224),**kwargs):
        super().__init__(*args,**kwargs)
        self.crop_size = crop_size
        
    def load(self,*args,**kwargs):
        return self.load_image(*args,**kwargs)



############################################
# Complete Loaders
############################################

@DATASETS.register_module()
class ObjectLoaderCompleteBase(Loader):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __getitem__(self, obj_id):
        obj = self.obj_infos.get(obj_id, None)
        if obj:
            pts = self.load_points(obj)
            return  pts
        else:
            raise ValueError(f'obj_id {obj_id} does not exist in obj_infos')
        


@DATASETS.register_module()
class ObjectLoaderCompleteNuscenes(Loader):

    def __init__(self,train,version,metadata_path,*args,**kwargs):
        metadata = load_nuscenes_metadata_split(metadata_path,version,train)
        super().__init__(*args,metadata=metadata,**kwargs)

    def __getitem__(self, obj_id):
        obj = self.obj_infos.get(obj_id, None)
        if obj:
            pts = self.load_points(obj)
            return  pts
        else:
            raise ValueError(f'obj_id {obj_id} does not exist in obj_infos')
        
@DATASETS.register_module()
class ObjectLoaderCompleteWaymo(Loader):

    def __init__(self,metadata_path,*args,**kwargs):
        metadata = load_metadata(metadata_path,use_fix=False)
        super().__init__(*args,metadata=metadata,**kwargs)

    def __getitem__(self, obj_id):
        obj = self.obj_infos.get(obj_id, None)
        if obj:
            pts = self.load_points(obj)
            return  pts
        else:
            raise ValueError(f'obj_id {obj_id} does not exist in obj_infos')

@DATASETS.register_module()
class FakeCompleteLoader():
    def __init__(self,subsample_num=2048):
        self.ssn = subsample_num

    def __getitem__(self,tok):
        return np.zeros((3,self.ssn,))




