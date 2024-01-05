
import torch
from mmdet3d.models import TRACKERS



class FeatureSet():
    pass


@TRACKERS.register_module()
class PointFeatureSet(FeatureSet):

    def __init__(self,replace_all):
        super().__init__()
        self.set_attributes()
        self.replace_all = replace_all


    def set_attributes(self):
        self.pts_feats = None
        self.pts_xyz = None
        self.lengths = None


    def reset(self):
        del self.pts_feats
        del self.pts_xyz
        del self.lengths
        self.set_attributes()


    def get_features(self,index):
        return self.pts_xyz[index,...],self.pts_feats[index,...]
    
    def store_new(self,xyz,feats,lengths):
        if self.pts_feats is None:
            self.pts_feats = feats
            self.pts_xyz = xyz
            self.lengths = lengths
        else:
            self.pts_feats = torch.cat([self.pts_feats,feats],dim=0)
            self.pts_xyz = torch.cat([self.pts_xyz,xyz],dim=0)
            self.lengths = torch.cat([self.lengths,lengths],dim=0)


    def replace_old(self,index,xyz,feats,lengths):
        """Only replace the old features if the new ones are longer"""
        
        if self.pts_feats is None:
            raise ValueError('pts_feats should not be None for replace_old -- there is a bug allowing tracks without there being detections')


        if self.replace_all:
            self.pts_feats[index,...] = feats
            self.pts_xyz[index,...] = xyz
            self.lengths[index,...] = lengths
        else:
            replace = torch.where(self.lengths[index] <= lengths)[0]
            index = index[replace]
            self.pts_feats[index,...] = feats[replace]
            self.pts_xyz[index,...] = xyz[replace]
            self.lengths[index,...] = lengths[replace]




@TRACKERS.register_module()
class ImageFeatureSet(FeatureSet):

    def __init__(self,replace_all):
        super().__init__()
        self.set_attributes()
        self.replace_all = replace_all

    def set_attributes(self):
        self.im_feats = None
        self.vis_level = None

    def reset(self):
        del self.im_feats
        del self.vis_level
        self.set_attributes()

    def get_features(self,index):
        return self.im_feats[index,...]
    
    def store_new(self,feats,vis_level):
        if self.im_feats is None:
            self.im_feats = feats
            self.vis_level = vis_level
        else:
            self.im_feats = torch.cat([self.im_feats,feats],dim=0)
            self.vis_level = torch.cat([self.vis_level,vis_level],dim=0)

    def replace_old(self,index,feats,vis_level):
        """Only replace the old features if the new ones are longer"""
        
        if self.im_feats is None:
            raise ValueError('im_feats should not be None for replace_old -- there is a bug allowing tracks without there being detections')


        if self.replace_all:
            self.im_feats[index,...] = feats
            self.vis_level[index,...] = vis_level
        else:
            replace = torch.where(self.vis_level[index] <= vis_level)[0]
            index = index[replace]
            self.im_feats[index,...] = feats[replace]
            self.vis_level[index,...] = vis_level[replace]



