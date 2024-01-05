import copy
import torch
# import pytorch3d

import torch.nn as nn
import numpy as np

from mmdet3d.models import TRACKERS
from mmdet3d.models  import builder
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS, build_iou_calculator
from mmdet3d.core.bbox.iou_calculators import BboxOverlapsNearest3D
from mmdet3d.ops import points_in_boxes_batch
from mmdet3d.core import LiDARInstance3DBoxes

from .track import Track
from .tracking_helpers import get_bbox_sides_and_center, EmptyBBox, get_cost_mat_viz, log_mistakes
from .transforms import affine_transform, apply_transform

from telnetlib import PRAGMA_HEARTBEAT
from functools import reduce
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
# from pytorch3d.structures.pointclouds import Pointclouds

from mmdet3d.models.trackers.pc_utils import (interpolate_per_frame, get_input_batch)




@IOU_CALCULATORS.register_module()
class Center2DRange(object):
    """Within 2D range"""

    def __init__(self, distance=2, coordinate='lidar'):
        # assert coordinate in ['camera', 'lidar', 'depth']
        self.distance = distance

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """assumes xy are the first two coordinates"""
        iou = torch.cdist(bboxes1[:,:2],bboxes2[:,:2],p=2) 
        # idx1,idx2 = torch.where(iou < self.distance)
        return iou #, idx1, idx2

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


@TRACKERS.register_module()
class VirtualTracker(nn.Module):
    ioucal = BboxOverlapsNearest3D('lidar')


    def log_str(self,timestep=None):
        if timestep == None:
            return "[VirtualTracker | Rank:{}]".format(self.rank)
        else:
            return "[VirtualTracker | Rank:{}, timestep: {}]".format(self.rank,timestep)

    def __init__(self, 
                 cls,
                 frameLimit=10,
                 tp_threshold=0.01,
                 rank=-1, 
                 track_supervisor=None, 
                 gt_testing=False,
                 box_output_mode='det',
                 use_nms=True,
                 suppress_threshold=0.15,
                 associator=None,
                 incrementor=None, 
                 modifier=None, 
                 updater=None,
                 point_reidentifier=None,
                 image_reidentifier=None,
                 verbose=False,
                 use_mp_feats=True,
                 detection_decisions=['det_false_positive','det_newborn'], 
                 tracking_decisions=[],
                 teacher_forcing=False,
                 ioucal=dict(type='BboxOverlapsNearest3D', coordinate='lidar'),
                 visualize_cost_mat=False,
                 propagation_method='velocity', # 'velocity' or 'future'
                 ):
        """The Track Manager implements the logic for tracking objects of the same class

        args:
            cls (string): class name
            frameLimit (int): The number of frames to persist unmatched tracks 
            tp_threshold (float): tracking score threshold for true positives
            rank (int): rank of the process
        
        """
        
        super().__init__()

        self.cls = cls
        self.tp_threshold = tp_threshold
        self.verbose = verbose
        self.rank = rank
        self.frameLimit = frameLimit
        self.suppress_threshold = suppress_threshold
        self.use_nms = use_nms
        self.teacher_forcing = teacher_forcing
        self.visualize_cost_mat = visualize_cost_mat
        self.ioucal = build_iou_calculator(ioucal)
        self.box_output_mode = box_output_mode
        self.gt_testing = gt_testing
        self.use_mp_feats = use_mp_feats #update detection features with their representation after transformer
        self.propagation_method = propagation_method
        self.epoch = 0 
        self.detection_decisions = sorted(detection_decisions)
        self.tracking_decisions = sorted(tracking_decisions) #need to sort for canonical ordering
        self.dd_num = len(detection_decisions)
        self.td_num = len(tracking_decisions)

        self.set_attributes(device=None)
        
        self.tsup = builder.build_supervisor(track_supervisor)
        self.associator = builder.build_tracker(associator)
        self.incrementor = builder.build_tracker(incrementor)
        self.modifier = builder.build_tracker(modifier)
        self.updater = builder.build_tracker(updater)
        self.point_reidentifier = builder.build_tracker(point_reidentifier)
        self.image_reidentifier = builder.build_tracker(image_reidentifier)

        self.matching_metrics = builder.build_tracker(dict(type='MatchingMetrics'))
        # self.point_reidentifier.point_feature_set = builder.build_tracker(dict(type='PointFeatureSet'))


    def reset(self,net,device):
        """Resets the TrackManager to default for the next sequence"""
        del self.tracks
        del  self.activeTracks
        del  self.unmatchedTracks
        del  self.decomTracks
        del self.trkid_to_gt_trkid
        del self.trkid_to_gt_tte
        self.incrementor.reset(net,device)
        self.point_reidentifier.reset()
        del self.logging
        del self.mistakes_track
        del self.mistakes_det
        self.set_attributes(device)

    def set_attributes(self,device):
        self.tracks = []
        self.activeTracks = []
        self.unmatchedTracks = {}
        self.decomTracks = []
        if device is not None:
            self.trkid_to_gt_trkid = torch.empty((0,),dtype=torch.long,device=device)
            self.trkid_to_gt_tte = torch.empty((0,),dtype=torch.long,device=device)
        else:
            self.trkid_to_gt_trkid = None
            self.trkid_to_gt_tte = None
        self.logging = {}
        for k in ['total','det_match'] + self.detection_decisions + self.tracking_decisions:
            self.logging[k+'_correct'] = 0
            self.logging[k+'_gt'] = 0.00000000001 # avoid divide by zero
            self.logging[k+'_num_pred'] = 0.00000000001 # avoid divide by zero  

        self.mistakes_track = {k:dict() for k in self.tracking_decisions + ['match']}
        self.mistakes_det = {k:dict() for k in self.detection_decisions + ['match']}
        self.mistakes_match = {}

    def set_epoch(self,epoch,max_epoch):
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.modifier.set_epoch(epoch,max_epoch)


    def gatherAny(self,feats,index):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        return self.incrementor.gatherAny(feats,index)

    def gatherActive(self,feats):
        """Indexes into the appropriate tensor to gather the desired
        freatures."""
        return self.incrementor.gatherAny(feats,self.activeTracks)


    def get_iou_idx(self,bbox,gt_bboxes,device,cls1=None,cls2=None,):
        if cls1 != None and cls2 != None and len(cls1) > 0 and len(cls2) > 0:
            # print(cls1.shape)
            # print(cls2.shape)
            # print(bbox.tensor.shape)
            # print(gt_bboxes.tensor.shape)
            assert len(cls1) == bbox.tensor.size(0)
            assert len(cls2) == gt_bboxes.tensor.size(0)
            mask = torch.zeros((bbox.tensor.size(0),gt_bboxes.tensor.size(0),),dtype=torch.float32,device=device)
            cp = torch.cartesian_prod(torch.arange(0,bbox.tensor.size(0)), torch.arange(0,gt_bboxes.tensor.size(0)))
            mask[cp[:,0],cp[:,1]] = (cls1.long()[cp[:,0]] != cls2.long()[cp[:,1]]).float() * 10000.0
        else:
            mask = torch.zeros((bbox.tensor.size(0),gt_bboxes.tensor.size(0),),dtype=torch.float32,device=device)
            

        if type(self.ioucal) == Center2DRange:

            det_iou = self.ioucal(bbox.tensor,gt_bboxes.tensor)
            det_iou = det_iou + mask

            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)
        
            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < self.ioucal.distance)
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]

        else:
            det_iou = self.ioucal(bbox.tensor,gt_bboxes.tensor) * -1 
            det_iou = det_iou + mask
            # det_iou[torch.where(det_iou > ( self.tp_threshold * -1) )] = 10000.
            tp_det_idx, tp_gt_idx = linear_sum_assignment(det_iou.cpu().numpy())
            tp_det_idx = torch.from_numpy(tp_det_idx).to(device)
            tp_gt_idx = torch.from_numpy(tp_gt_idx).to(device)

            matches = torch.where(det_iou[tp_det_idx,tp_gt_idx] < ( self.tp_threshold * -1))
            tp_det_idx = tp_det_idx[matches]
            tp_gt_idx = tp_gt_idx[matches]

        


        return det_iou[tp_det_idx,tp_gt_idx], tp_det_idx, tp_gt_idx


    def non_max_suppression(self,device):
        """nms on the active tracks"""
        if not self.use_nms:
            return

        if len(self.tracks) == 0 or len(self.activeTracks) == 0:
            return

        classes = torch.tensor([self.tracks[i].cls[-1] for i in self.activeTracks],dtype=torch.long, device=device)
        mask = torch.zeros((len(self.activeTracks),len(self.activeTracks)),dtype=torch.float32, device=device)
        cp = torch.cartesian_prod(torch.arange(0,len(self.activeTracks)), torch.arange(0,len(self.activeTracks))).to(device)
        mask[cp[:,0],cp[:,1]] = (classes[cp[:,0]] != classes[cp[:,1]]).float() * -10000.0

        bboxes = torch.stack([self.tracks[i].det_bboxes[-1] for i in self.activeTracks], dim=0)
        iou = VirtualTracker.ioucal(bboxes,bboxes)
        iou = iou + mask

        iou = torch.triu(iou, diagonal=1) # suppress duplicate entires 
        idx1,idx2 = torch.where(iou > self.suppress_threshold)
        # print('to_suppress: ',idx1,idx2)

        track_score = torch.stack([len(self.tracks[i].det_bboxes) + self.tracks[i].scores[-1] for i in self.activeTracks], dim=0)
        scores = track_score[idx1] - track_score[idx2]
        suppress = [self.activeTracks[x] for x in idx1[torch.where(scores <= 0)]] + [self.activeTracks[x] for x in idx2[torch.where(scores > 0)]]
        # print("[actually suppressing]",suppress)

        self.activeTracks = [x for x in self.activeTracks if x not in suppress]
        self.decomTracks += suppress


    def get_dets_to_trk_idx(self,num_dets,active_tracks,decisions,device):
        """Returns a tensor mapping each incoming detection to its global track ID. All 
        detections are kept, but not all are made into active tracks."""
        dets_to_trk_idx = torch.zeros(num_dets,dtype=torch.int64,device=device)
        dets_to_trk_idx[decisions['det_match']] = torch.tensor(active_tracks,dtype=torch.long,device=device)[decisions['track_match']]
        offset = 0
        for k in self.detection_decisions:
            
            dets_to_trk_idx[decisions[k]] = torch.tensor([len(self.tracks) + offset + ii  for ii in range(len(decisions[k]))], #assign new track indices to new dets
                                                            dtype=torch.long,
                                                            device=device)
            offset += len(decisions[k])

        return dets_to_trk_idx

    def get_active_bboxes(self,timestep):
        try:
            return [self.tracks[i][timestep] for i in self.activeTracks]
        except KeyError:
            print("Rank:",self.rank)
            print("timestep:",timestep)
            print("active_tracks",self.activeTracks)
            exit(0)


    def get_track_det_distances(self,bbox,ego,timestep,device):
        if bbox.tensor.size(0) == 0 or len(self.activeTracks) == 0:
            return []

        dets_xyz = bbox.tensor[:,:3]
        dets_prev = ego.transform_over_time(dets_xyz.cpu().numpy(),from_=timestep,to_=timestep-1)
        track_xyz = torch.stack(self.get_active_bboxes(timestep=timestep-1))[:,:2]
        dists = torch.cdist(track_xyz,torch.from_numpy(dets_prev.astype(np.float32)).to(device)[:,:2],p=2.0)
        return dists
        
    def update_gt_track_mapping(self,gt_track_tte,gt_tracks,tp_det_idx,tp_gt_idx,dets_to_trk_idx,device):
        """Updates the mapping from track ID to GT track ID and GT track TTE

        Takes indices of the kept detections at the current timestep
        and stores them at index positions correspoiinding to their track 
        id in tensors for fast lookup later.

        Args:
            gt_track_tte (torch.Tensor): tensor of GT track TTEs.
            gt_tracks (torch.Tensor): tensor of GT track IDs.
            tp_det_idx (torch.Tensor): indices of TP detections corresponding to tp_gt_idx.
            tp_gt_idx (torch.Tensor): indices of GT tracks corresponding to TP detections.
            dets_to_trk_idx (torch.Tensor): tensor mapping each detection to its global track ID.
            device (torch.device): device to store tensors on.
        
        """
        #-2: newborn -1: false positive
        update_trkid = torch.full((self.incrementor.track_feats.size(0),),-2,device=device)
        update_tte = torch.full((self.incrementor.track_feats.size(0),),-1,device=device)

        update_trkid[:self.trkid_to_gt_trkid.size(0)] = self.trkid_to_gt_trkid
        update_tte[:self.trkid_to_gt_tte.size(0)] += self.trkid_to_gt_tte #decrement one timestep

        #get False Positive detections
        temp = torch.full_like(dets_to_trk_idx,1)
        temp[tp_det_idx] = 0
        fp_det_idx = torch.where(temp == 1)[0]
        
        try:
            assert ( len(fp_det_idx) + len(tp_det_idx) ) == len(dets_to_trk_idx)
        except AssertionError:
            print("fp_det_idx:",len(fp_det_idx),fp_det_idx)
            print("tp_det_idx:",len(tp_det_idx),tp_det_idx)
            print("dets_to_trk_idx:",len(dets_to_trk_idx),dets_to_trk_idx)
            print("Temp",temp.shape)
            print("len(temp) - len(tp_det_idx) - len(fp_det_idx[0])",len(temp) - len(tp_det_idx) - len(fp_det_idx[0]))
            exit(0)

        #Update GT IDs of TP matches and newborns
        update_trkid[dets_to_trk_idx[tp_det_idx]] = gt_tracks[tp_gt_idx]
        update_tte[dets_to_trk_idx[tp_det_idx]] = gt_track_tte[tp_gt_idx]

        #Update GT IDs of FP matches
        update_trkid[dets_to_trk_idx[fp_det_idx]] = -1
        update_tte[dets_to_trk_idx[fp_det_idx]] = -1

        #set -2 to -1
        update_trkid[update_trkid == -2] = -1
 
        self.trkid_to_gt_trkid = update_trkid
        self.trkid_to_gt_tte = update_tte


    def log_reid_matching_score(self,tp_decisions,reid_cost_mat,reid_cp,num_tracks,num_dets,prefix,device):
        pos = reid_cost_mat[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']]
        mask = torch.zeros(num_tracks,num_dets,device=device)
        mask[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']] = -1
        try:
            mask[reid_cp[:,0],reid_cp[:,1]] -= 1 
        except TypeError:
            print('mask',mask)
            print('reid_cp',reid_cp)
            exit(0)

        neg_idx = torch.where(mask == -2)
        neg = reid_cost_mat[neg_idx]

        match_preds = torch.cat([pos,neg],dim=0)
        match_gt = torch.cat([torch.ones_like(pos),torch.zeros_like(neg)],dim=0)
        match_num = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match_gt).float()
        match_num_pos = match_num[:len(pos)]
        match_num_neg = match_num[len(pos):]
        match_acc = match_num.mean().item()
        self.logging[f'{prefix}_match_num'] = [torch.cat(self.logging.get(f'{prefix}_match_num',[])  + [match_num],dim=0)]
        self.logging[f'{prefix}_match_num_pos'] = [torch.cat(self.logging.get(f'{prefix}_match_num_pos',[])  + [match_num_pos],dim=0)]
        self.logging[f'{prefix}_match_num_neg'] = [torch.cat(self.logging.get(f'{prefix}_match_num_neg',[])  + [match_num_neg],dim=0)]

        

        if len(pos) > 0 and len(neg) >0:
            match_lengths_pos = torch.cat([pts_lengths_track[tp_decisions[f'{prefix}_pos_track_match']].unsqueeze(1),pts_lengths_det[tp_decisions[f'{prefix}_pos_det_match']].unsqueeze(1)],dim=1).min(1).values
            match_lengths_neg = torch.cat([pts_lengths_track[neg_idx[0]].unsqueeze(1),pts_lengths_det[neg_idx[1]].unsqueeze(1)],dim=1).min(1).values
            match_lengths = torch.cat([match_lengths_pos,match_lengths_neg],dim=0)

            assert match_lengths_pos.size(0) == len(pos)
            assert match_lengths_neg.size(0) == len(neg)
            assert match_lengths.size(0) == len(pos) + len(neg)

            self.logging[f'{prefix}_match_min_length'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length',[])  + [match_lengths],dim=0)]
            self.logging[f'{prefix}_match_min_length_pos'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length_pos',[])  + [match_lengths_pos],dim=0)]
            self.logging[f'{prefix}_match_min_length_neg'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length_neg',[])  + [match_lengths_neg],dim=0)]
        elif len(pos) == 0 and len(neg) > 0:
            match_lengths_neg = torch.cat([pts_lengths_track[neg_idx[0]].unsqueeze(1),pts_lengths_det[neg_idx[1]].unsqueeze(1)],dim=1).min(1).values
            match_lengths = match_lengths_neg

            self.logging[f'{prefix}_match_min_length'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length',[])  + [match_lengths],dim=0)]
            self.logging[f'{prefix}_match_min_length_neg'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length_neg',[])  + [match_lengths_neg],dim=0)]
        elif len(pos) > 0 and len(neg) == 0:
            match_lengths_pos = torch.cat([pts_lengths_track[tp_decisions['pos_track_match']].unsqueeze(1),pts_lengths_det[tp_decisions['pos_det_match']].unsqueeze(1)],dim=1).min(1).values
            match_lengths = match_lengths_pos

            self.logging[f'{prefix}_match_min_length'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length',[])  + [match_lengths],dim=0)]
            self.logging[f'{prefix}_match_min_length_pos'] = [torch.cat(self.logging.get(f'{prefix}_match_min_length_pos',[])  + [match_lengths_pos],dim=0)]
        else:
            pass




        mask = torch.zeros(num_tracks,num_dets,device=device)
        mask[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = match_num_pos

        correct = torch.where(mask == 1)

        mask2 = torch.zeros(num_tracks,num_dets,device=device)
        mask2[decisions['track_match'],decisions['det_match']] = 1


        reid_corr_only = torch.where(mask2[correct] == 0)[0]

        self.logging[f'{prefix}_reid_correct_only'] = self.logging.get(f'{prefix}_reid_correct_only',[])  + [reid_corr_only.size(0)]
        self.logging[f'{prefix}_reid_correct'] = self.logging.get(f'{prefix}_reid_correct',[])  + [correct[0].size(0)]


        # reid_corr_only = torch.where(mask == 1)[0]
        if correct[0].size(0) > 0:
            print("[Tracker] matches where reid was the only one correct {}, percentage of correct reid where tracker was wrong {:.2f}% ({}/{})".format(reid_corr_only.size(0),
                                                                                                                                                (reid_corr_only.size(0)/correct[0].size(0)) * 100,
                                                                                                                                                reid_corr_only.size(0),correct[0].size(0)))
            self.logging[f'{prefix}_reid_correct_only'] = self.logging.get(f'{prefix}_reid_correct_only',[])  + [reid_corr_only.size(0)]
            self.logging[f'{prefix}_reid_correct'] = self.logging.get(f'{prefix}_reid_correct',[])  + [correct[0].size(0)]
                                                                                                                                            
        else:
            print("[Tracker] 0 matches where reid was correct")

    
    def step(self,
             ego,
             net,
             timestep,
             points,
             pred_cls,
             bev_feats,
             det_feats,
             point_cloud_range,
             bbox,
             trackCount,
             device,
             last_in_scene,
             det_confidence,
             zero_feats,
             orig_imgs=None,
             camera_intrinsics=None,
             lidar2camera=None,
             sample_token=None,
             sweeps_list=None,
             gt_labels=None,
             gt_bboxes=[],
             gt_tracks=None,
             output_preds=False,
             return_loss=True,
             gt_futures=None,
             gt_pasts=None,
             gt_track_tte=None):
        """Take one step forward in time by processing the current frame.

        Args:
            det_feats (torch.Tensor, required): Feature representation 
                for the detections computed by MLPMerge
            bbox (torch.Tensor, required): bounding boxes for each 
                detection
            lstm (nn.LSTM, required): lstm for processing tracks
            trackCount (int, required): the number of tracks currently
                allocated for this scene
            c0 (torch.Tensor): lstm initial cell state
            h0 (torch.Tensor): lstm initial hidden state
            sample_token (str, optional): token of the current detection
                used for computing performance metrics
            gt_bboxes (LiDARInstance3DBoxes): GT bounding boxes
            gt_tracks (torch.Tensor): GT tracks ids
            output_preds (bool): If True outputs tracking predictions o/w None
            return_loss (bool): If True compute the loss
        
        Returns: 
            trackCount (int): the new track count
            log_vars (dict): information to be logged 
            losses (dict): losses computed from the association matrix
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if output_preds is True, None otherwise

        """
        if self.verbose:
            print("{} entering step().".format(self.log_str(timestep)))
            print("Timestep:{}, num_tracks:{}, num_dets:{}, rank:{}".format(timestep,num_tracks,num_dets,self.rank))


        assert timestep >= 0
        log_vars = {}
        num_tracks = len(self.activeTracks)
        num_dets = det_feats.size(0)

        # add info to scene level logger
        self.log_update({'mean_num_tracks_per_timestep':torch.tensor([num_tracks],dtype=torch.float32),
                         'mean_num_dets_per_timestep':torch.tensor([num_dets],dtype=torch.float32)})
        
        trk_feats = self.gatherActive(['f'])
        trk_feats = trk_feats.clone()
        decision_feats = torch.tensor([num_tracks,num_dets],dtype=torch.float32,device=device)



        pts_cost_mat, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp = \
                self.point_reidentifier(net=net,
                                        tracker=self,
                                        num_tracks=num_tracks,
                                        num_dets=num_dets,
                                        sweeps_list=sweeps_list,
                                        pred_cls=pred_cls,
                                        bbox=bbox,
                                        device=device)
        


        im_cost_mat, im_feats_det, vis_det, im_feats_track, vis_track, im_cp = \
                self.image_reidentifier(net=net,
                                        tracker=self,
                                        pred_cls=pred_cls,
                                        orig_imgs=orig_imgs,
                                        camera_intrinsics=camera_intrinsics,
                                        lidar2camera=lidar2camera,
                                        boxes_3d=bbox.tensor[:,:7],
                                        num_tracks=num_tracks,
                                        num_dets=num_dets,
                                        device=device,)


        supervise, trk_feats_mp, det_feats_mp = \
                net.forward_association(trk_feats=trk_feats,
                                        det_feats=det_feats,
                                        decision_feats=decision_feats,
                                        pts_cost_mat=None,#pts_cost_mat,
                                        tracking_decisions=self.tracking_decisions,
                                        detection_decisions=self.detection_decisions,
                                        device=device,)


        if self.use_mp_feats:
            #increment using message passing features
            trk_feats = trk_feats_mp
            det_feats = det_feats_mp


        if self.associator.use_distance_prior:
            track_det_dists = self.get_track_det_distances(bbox=bbox,
                                                           ego=ego,
                                                           timestep=timestep,
                                                           device=device) 
        else:
            track_det_dists = []

        
        class_mask = torch.zeros((num_tracks,num_dets),dtype=torch.float32,device=device)
        track_classes = torch.tensor([self.tracks[x].cls[-1] for x in self.activeTracks],dtype=torch.long,device=device)
        cp = torch.cartesian_prod(torch.arange(0,num_tracks), torch.arange(0,num_dets))
        class_mask[cp[:,0],cp[:,1]] = (track_classes[cp[:,0]] != pred_cls[cp[:,1]]).float() * 10000.0

        decisions, summary, (numpy_cost_mat, dd_mat, td_mat) = self.associator(supervise,
                                                                                tracker=self, 
                                                                                num_trk=num_tracks, 
                                                                                num_det=num_dets, 
                                                                                class_name=self.cls, 
                                                                                track_det_dists=track_det_dists, 
                                                                                class_mask=class_mask,
                                                                                device=device)
        log_vars.update(summary)


        if gt_tracks != None:
            tp_decisions, mod_decisions, tp_det_idx, tp_gt_idx  = self.modifier(tracker=self,
                                                                                bbox=bbox,
                                                                                gt_bboxes=gt_bboxes,
                                                                                gt_tracks=gt_tracks,
                                                                                pred_cls=pred_cls,
                                                                                gt_labels=gt_labels,
                                                                                prev_AT=self.activeTracks,
                                                                                decisions=decisions,
                                                                                device=device)

            self.log_update({f"num_TP_{k[4:]}":torch.tensor([len(v)],dtype=torch.float32) for k,v in tp_decisions.items() if k.startswith('pos')})
            self.log_update({f"num_dec_{k}":torch.tensor([len(v)],dtype=torch.float32) for k,v in decisions.items()})

            #contains asserts
            self.verify_decisions(decisions,tp_decisions,gt_tracks,gt_track_tte,timestep,num_dets,num_tracks)

            #save visualizations of the cost matrix
            if self.visualize_cost_mat:
                get_cost_mat_viz(tracker=self,
                                tp_decisions=tp_decisions,
                                decisions=decisions,
                                td_mat=td_mat,
                                dd_mat=dd_mat,
                                cost_mat=numpy_cost_mat,
                                num_track=num_tracks,
                                num_det=num_dets,
                                device=device,
                                save_dir='/mnt/bnas/tracking_results/cost_mats/{}.pdf'.format(sample_token),
                                show=False)
                

        if return_loss == False:
            log_mistakes(tracker=self,
                        tp_decisions=tp_decisions,
                        decisions=decisions,
                        td_mat=td_mat,
                        dd_mat=dd_mat,
                        cost_mat=numpy_cost_mat,
                        num_track=num_tracks,
                        num_det=num_dets,
                        det_tp_idx=tp_det_idx,
                        active_tracks=self.activeTracks,
                        device=device,)
            

        if ( return_loss == True and self.teacher_forcing ) or self.gt_testing:
            #teacher forcing
            decisions = mod_decisions

        if pts_cost_mat is not None and im_cost_mat is not None:
            # print("Here")
            self.matching_metrics.log_match_preds(tracker=self,
                                                tp_decisions=tp_decisions,
                                                decisions=decisions,
                                                pts_cost_mat=pts_cost_mat,
                                                im_cost_mat=im_cost_mat,
                                                pts_lengths_track=pts_lengths_track,
                                                pts_lengths_det=pts_lengths_det,
                                                device=device)



        # self.log_reid_matching_score(tp_decisions=tp_decisions,
        #                              reid_cost_mat=pts_cost_mat,
        #                              reid_cp=pts_cp,
        #                              num_tracks=num_tracks,
        #                              num_dets=num_dets,
        #                              prefix='pts',
        #                              device=device,)
        



        # if pts_cost_mat is not None \
        #     and self.point_reidentifier.use_pc_feats \
        #     and pts_cp is not None \
        #     and return_loss == False:


            # pos = pts_cost_mat[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']]
            # mask = torch.zeros(num_tracks,num_dets,device=device)
            # mask[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']] = -1
            # try:
            #     mask[pts_cp[:,0],pts_cp[:,1]] -= 1 #mask[pts_cp[:,0],pts_cp[:,1]]  - 1
            # except TypeError:
            #     print('mask',mask)
            #     print('pts_cp',pts_cp)
            #     exit(0)

            # neg_idx = torch.where(mask == -2)
            # neg = pts_cost_mat[neg_idx]

            # match_preds = torch.cat([pos,neg],dim=0)
            # match_gt = torch.cat([torch.ones_like(pos),torch.zeros_like(neg)],dim=0)
            # match_num = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match_gt).float()
            # match_num_pos = match_num[:len(pos)]
            # match_num_neg = match_num[len(pos):]
            # match_acc = match_num.mean().item()
            # self.logging['match_num'] = [torch.cat(self.logging.get('match_num',[])  + [match_num],dim=0)]
            # self.logging['match_num_pos'] = [torch.cat(self.logging.get('match_num_pos',[])  + [match_num_pos],dim=0)]
            # self.logging['match_num_neg'] = [torch.cat(self.logging.get('match_num_neg',[])  + [match_num_neg],dim=0)]

            # if len(pos) > 0 and len(neg) >0:
            #     match_lengths_pos = torch.cat([pts_lengths_track[tp_decisions['pos_track_match']].unsqueeze(1),pts_lengths_det[tp_decisions['pos_det_match']].unsqueeze(1)],dim=1).min(1).values
            #     match_lengths_neg = torch.cat([pts_lengths_track[neg_idx[0]].unsqueeze(1),pts_lengths_det[neg_idx[1]].unsqueeze(1)],dim=1).min(1).values
            #     match_lengths = torch.cat([match_lengths_pos,match_lengths_neg],dim=0)

            #     assert match_lengths_pos.size(0) == len(pos)
            #     assert match_lengths_neg.size(0) == len(neg)
            #     assert match_lengths.size(0) == len(pos) + len(neg)

            #     self.logging['match_min_length'] = [torch.cat(self.logging.get('match_min_length',[])  + [match_lengths],dim=0)]
            #     self.logging['match_min_length_pos'] = [torch.cat(self.logging.get('match_min_length_pos',[])  + [match_lengths_pos],dim=0)]
            #     self.logging['match_min_length_neg'] = [torch.cat(self.logging.get('match_min_length_neg',[])  + [match_lengths_neg],dim=0)]
            # elif len(pos) == 0 and len(neg) > 0:
            #     match_lengths_neg = torch.cat([pts_lengths_track[neg_idx[0]].unsqueeze(1),pts_lengths_det[neg_idx[1]].unsqueeze(1)],dim=1).min(1).values
            #     match_lengths = match_lengths_neg

            #     self.logging['match_min_length'] = [torch.cat(self.logging.get('match_min_length',[])  + [match_lengths],dim=0)]
            #     self.logging['match_min_length_neg'] = [torch.cat(self.logging.get('match_min_length_neg',[])  + [match_lengths_neg],dim=0)]
            # elif len(pos) > 0 and len(neg) == 0:
            #     match_lengths_pos = torch.cat([pts_lengths_track[tp_decisions['pos_track_match']].unsqueeze(1),pts_lengths_det[tp_decisions['pos_det_match']].unsqueeze(1)],dim=1).min(1).values
            #     match_lengths = match_lengths_pos

            #     self.logging['match_min_length'] = [torch.cat(self.logging.get('match_min_length',[])  + [match_lengths],dim=0)]
            #     self.logging['match_min_length_pos'] = [torch.cat(self.logging.get('match_min_length_pos',[])  + [match_lengths_pos],dim=0)]
            # else:
            #     pass


            # mask = torch.zeros(num_tracks,num_dets,device=device)
            # mask[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']] = match_num_pos

            # correct = torch.where(mask == 1)

            # mask2 = torch.zeros(num_tracks,num_dets,device=device)
            # mask2[decisions['track_match'],decisions['det_match']] = 1


            # reid_corr_only = torch.where(mask2[correct] == 0)[0]

            # self.logging['reid_correct_only'] = self.logging.get('reid_correct_only',[])  + [reid_corr_only.size(0)]
            # self.logging['reid_correct'] = self.logging.get('reid_correct',[])  + [correct[0].size(0)]

            # if correct[0].size(0) > 0:
            #     print("[Tracker] matches where reid was the only one correct {}, percentage of correct reid where tracker was wrong {:.2f} ({}/{})".format(reid_corr_only.size(0),
            #                                                                                                                                         (reid_corr_only.size(0)/correct[0].size(0)) * 100,
            #                                                                                                                                         reid_corr_only.size(0),correct[0].size(0)))
            #     self.logging['reid_correct_only'] = self.logging.get('reid_correct_only',[])  + [reid_corr_only.size(0)]
            #     self.logging['reid_correct'] = self.logging.get('reid_correct',[])  + [correct[0].size(0)]
                                                                                                                                                
            # else:
            #     print("[Tracker] 0 matches where reid was correct")





        
        
        if 'track_false_negative' in self.tracking_decisions:

            if len(decisions['track_false_negative'] ) > 0:
                # print(gt_bboxes.__dict__)

                bbox_list_fn = [torch.tensor(self.tracks[self.activeTracks[t]].transform_over_time(from_=timestep-1,to_=timestep,ego=ego,propagation_method=self.propagation_method)[1],
                                          dtype=torch.float32,
                                          device=device) # [0] for refined [1] for det
                                for t in decisions['track_false_negative'] ]
                bbox_list_fn = torch.stack(bbox_list_fn)
                bbox_list_fn = LiDARInstance3DBoxes(bbox_list_fn, box_dim=9, with_yaw=True, origin=(0.5, 0.5, 0))

                confidence_scores_fn = torch.stack([ self.tracks[self.activeTracks[t]].det_confidence[-1] for t in decisions['track_false_negative'] ])
                pred_cls_fn = torch.stack([self.tracks[self.activeTracks[t]].cls[-1] for t in decisions['track_false_negative'] ])

                false_negative_emb, _, _ = net.getMergedFeats(
                        ego=ego,
                        pred_cls=pred_cls_fn,
                        bbox_feats=bbox_list_fn.tensor.to(device),
                        bbox_side_and_center=get_bbox_sides_and_center(bbox_list_fn),
                        queries=net.synthetic_query(trk_feats[decisions['track_false_negative'],:]),
                        bev_feats=bev_feats,
                        confidence_scores=confidence_scores_fn,
                        point_cloud_range=point_cloud_range,
                        zero_feats=zero_feats,
                        im_crops=torch.zeros((len(decisions['track_false_negative']),3,224,224),device=device),
                        pts_crops=None,
                        device=device,
                    )


                if self.use_mp_feats:
                    false_negative_emb = false_negative_emb + trk_feats[decisions['track_false_negative'],:]


                    
                # false_negative_emb = net.false_negative_emb.weight.repeat(decisions['track_false_negative'].size(0),1)
                false_negative_idx = det_feats.size(0) + torch.arange(0,decisions['track_false_negative'].size(0),device=device)
            else:
                false_negative_idx = torch.empty((0),dtype=torch.long,device=device)
                false_negative_emb = torch.empty((0,det_feats.size(1)),dtype=torch.float32,device=device)

            #take an LSTM step based on the new tracks
            self.incrementor(net=net,
                            active_tracks=self.activeTracks,
                            hidden_idx=torch.cat([decisions['track_match'],decisions['track_false_negative']]),
                            increment_hidden_idx=torch.cat([decisions['det_match'],false_negative_idx]),
                            new_idx=torch.cat([decisions[k] for k in self.detection_decisions]),
                            feats=torch.cat([det_feats,false_negative_emb],dim=0),
                            device=device)
        else:
            self.incrementor(net=net,
                            active_tracks=self.activeTracks,
                            hidden_idx=decisions['track_match'],
                            increment_hidden_idx=decisions['det_match'],
                            new_idx=torch.cat([decisions[k] for k in self.detection_decisions]),
                            feats=det_feats,
                            device=device)



        if self.point_reidentifier.use_pc_feats:

            if num_dets > 0:
                new_det_idx_ = torch.cat([decisions[k] for k in self.detection_decisions])
                self.point_reidentifier.point_feature_set.store_new(xyz_det[new_det_idx_,...], 
                                                                    pts_feat_det[new_det_idx_,...], 
                                                                    pts_lengths_det[new_det_idx_])
            
            if len(decisions['track_match']) > 0:
                self.point_reidentifier.point_feature_set.replace_old(index=decisions['track_match'],
                                                                      xyz=xyz_det[decisions['det_match'],...], 
                                                                      feats=pts_feat_det[decisions['det_match'],...],
                                                                      lengths=pts_lengths_det[decisions['det_match']])
                


        if self.image_reidentifier.use_im_feats:

            if num_dets > 0:
                new_det_idx_ = torch.cat([decisions[k] for k in self.detection_decisions])
                self.image_reidentifier.image_feature_set.store_new(feats=im_feats_det[new_det_idx_,...], 
                                                                    vis_level=vis_det[new_det_idx_])
            
            if len(decisions['track_match']) > 0:
                self.image_reidentifier.image_feature_set.replace_old(index=decisions['track_match'],
                                                                      feats=im_feats_det[decisions['det_match'],...], 
                                                                      vis_level=vis_det[decisions['det_match'],...],
                                            )






        #TODO find a better way to retain active tracks
        prev_AT = copy.deepcopy(self.activeTracks)
        dets_to_trk_idx = self.get_dets_to_trk_idx(det_feats.size(0), prev_AT, decisions, device)

        if gt_tracks != None:#train vs testing
            self.update_gt_track_mapping(gt_track_tte,gt_tracks,tp_det_idx,tp_gt_idx,dets_to_trk_idx,device)

                                                
        assert self.incrementor.track_feats.size(0) >= dets_to_trk_idx.size(0)

        # print(self.incrementor.track_feats.shape)
        current_trk_feats = self.gatherAny('f',dets_to_trk_idx)
        forecast_preds = net.MLPPredict(current_trk_feats)
        refine_preds = net.MLPRefine(current_trk_feats)
        refine_preds = torch.cat([refine_preds[:,:-1],torch.sigmoid(refine_preds[:,-1]).unsqueeze(1)],dim=1)

        #update the actual Track objects representing the tracks
        trackCount, tracks_to_output = self.updater(tracker=self,
                                                    pred_cls=pred_cls,
                                                    det_confidence=det_confidence,
                                                    trackCount=trackCount,
                                                    bbox=bbox,
                                                    decisions=decisions,
                                                    sample_token=sample_token,
                                                    refine_preds=refine_preds,
                                                    forecast_preds=forecast_preds,
                                                    timestep=timestep,
                                                    ego=ego,
                                                    device=device)




        assert len(self.tracks) >= dets_to_trk_idx.size(0)
        assert dets_to_trk_idx.size(0) == refine_preds.size(0) 

        losses = torch.tensor(0.,device=device, requires_grad=True)
        if return_loss: #only calculate loss info during training
            # print("In supervise tracking seciton")

            assert len(prev_AT) == trk_feats.size(0)
            assert dets_to_trk_idx.size(0) == det_feats.size(0)
            assert dets_to_trk_idx.size(0) == bbox.tensor.size(0)

            
            association_loss, summary, log = self.tsup.supervise_association(
                    tracker=self,
                    tp_decisions=tp_decisions,
                    supervise=supervise,
                    device=device,
                    return_loss=return_loss
            )
            log_vars.update(summary)
            self.log_update(log)


            if len(dets_to_trk_idx) == 0:
                refined_bboxes = torch.empty([0,9],dtype=torch.float32,device=device)
            else:
                refined_bboxes = torch.cat([self.tracks[i].refined_bboxes[-1].unsqueeze(0) for i in dets_to_trk_idx])

            refine_loss, summary, log = self.tsup.supervise_refinement(
                                            tp_det_idx=tp_det_idx,
                                            tp_gt_idx=tp_gt_idx,
                                            refine_preds=refine_preds,
                                            refined_bboxes=refined_bboxes,
                                            bbox=bbox,
                                            gt_pasts=gt_pasts,
                                            gt_bboxes=gt_bboxes,
                                            device=device,
                                            return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)

            forecast_loss, summary, log = self.tsup.supervise_forecast(
                                        tp_det_idx=tp_det_idx,
                                        tp_gt_idx=tp_gt_idx,
                                        forecast_preds=forecast_preds,
                                        gt_futures=gt_futures,
                                        device=device,
                                        log_prefix='track_',
                                        return_loss=return_loss)
            log_vars.update(summary)
            self.log_update(log)
            
            if gt_bboxes.tensor.nelement() == 0:
                #Naive workaround to avoid unused paramters error in torch
                refine_loss = torch.tensor(0.,device=device, requires_grad=True)# torch.tensor net.MLPRefine(torch.randn((10,net.MLPRefine[0].in_features),device=device)).sum() * 0.0
                forecast_loss = torch.tensor(0.,device=device, requires_grad=True)# net.MLPPredict(torch.randn((10,net.MLPPredict[0].in_features),device=device)).sum() * 0.0
                association_loss = association_loss * 0.0

            losses = refine_loss + forecast_loss + association_loss

        trk_id_preds, score_preds, bbox_preds, cls_preds = self.get_global_track_id_preds(tracks_to_output, device=device, output_preds=output_preds)

        if self.verbose:
            print("{} ending step().".format(self.log_str(timestep)))

        return trackCount, log_vars, losses, (trk_id_preds, score_preds, bbox_preds, cls_preds,)





    def log_update(self,x):
        for k in x: 
            try:
                self.logging[k].append(x[k])
            except KeyError:
                self.logging[k] = [x[k]]
        

    def get_scene_metrics(self,eval=False):
        metrics = {}
        if eval:
            # print(
            self.matching_metrics.eval_match(self.logging)
            # )
            # exit(0)
            if 'matching_metrics' in self.logging:
                metrics['eval_matching_metrics'] = self.matching_metrics.merge_match_metrics(self.logging)
                del self.logging['matching_metrics']

            # if self.point_reidentifier.use_pc_feats:
            #     print("self.logging",self.logging.keys())
            #     metrics['eval_match_num'] = self.logging['match_num'][0].cpu()
            #     metrics['eval_match_num_pos'] = self.logging['match_num_pos'][0].cpu()
            #     metrics['eval_match_num_neg'] = self.logging['match_num_neg'][0].cpu()
            #     metrics['eval_match_min_length'] =  self.logging['match_min_length'][0].cpu()
            #     metrics['eval_match_min_length_pos'] =  self.logging['match_min_length_pos'][0].cpu()
            #     metrics['eval_match_min_length_neg'] =  self.logging['match_min_length_neg'][0].cpu()  
            #     metrics['eval_reid_correct_only'] = torch.Tensor(self.logging['reid_correct_only']).cpu()
            #     metrics['eval_reid_correct'] = torch.Tensor(self.logging['reid_correct']).cpu()
            #     # print(metrics['eval_reid_correct'],metrics['eval_match_min_length_neg'])

            #     print("[Scene Metrics] ########################################################")
            #     print("[Scene Metrics] pts_match_acc: {:.2f}% ({}/{})".format(self.logging['match_num'][0].mean().item()*100,self.logging['match_num'][0].sum(),self.logging['match_num'][0].size(0)))
            #     print("[Scene Metrics] pts_match_acc_pos: {:.2f}% ({}/{})".format(self.logging['match_num_pos'][0].mean().item()*100,self.logging['match_num_pos'][0].sum(),self.logging['match_num_pos'][0].size(0)))
            #     print("[Scene Metrics] pts_match_acc_neg: {:.2f}% ({}/{})".format(self.logging['match_num_neg'][0].mean().item()*100,self.logging['match_num_neg'][0].sum(),self.logging['match_num_neg'][0].size(0)))

            #     print("[Scene Metrics] matches where reid was the only one correct {}, percentage of correct reid where tracker was wrong {:.2f} ({}/{})".format(
            #     np.sum(self.logging['reid_correct_only']),
            #     (np.sum(self.logging['reid_correct_only'])/np.sum(self.logging['reid_correct'])) * 100,
            #     np.sum(self.logging['reid_correct_only']),np.sum(self.logging['reid_correct'])))
            #     print("[Scene Metrics] ########################################################")

            #     del self.logging['match_num']
            #     del self.logging['match_num_pos']
            #     del self.logging['match_num_neg']
            #     del self.logging['match_min_length']
            #     del self.logging['match_min_length_pos']
            #     del self.logging['match_min_length_neg']
            #     del self.logging['reid_correct_only']
            #     del self.logging['reid_correct']

            metrics.update({'eval_'+ k:v for k,v in self.logging.items()})
            len_tracks = [len(trk.refined_bboxes) for trk in self.tracks]
            metrics[f'eval_len_tracks'] = len_tracks
            greater_than_one_tracks = [x for x in len_tracks if x > 1]
            metrics[f'eval_greater_than_one_tracks'] = greater_than_one_tracks
            metrics[f'eval_scene_total_tracks'] = len(self.tracks)
            metrics.update({'mistakes_track_'+ k:v for k,v in self.mistakes_track.items()})
            metrics.update({'mistakes_det_'+ k:v for k,v in self.mistakes_det.items()})
            metrics.update({'mistakes_match_'+ k:v for k,v in self.mistakes_match.items()})

            return metrics

        else:
            len_tracks = [len(trk.refined_bboxes) for trk in self.tracks]
            metrics[f'mean_track_length_{self.cls}'] = np.mean(len_tracks)
            metrics[f'median_track_length_{self.cls}'] = np.median(len_tracks)

            greater_than_one_tracks = [x for x in len_tracks if x > 1]
            if greater_than_one_tracks != []:
                metrics[f'mean_track_length_>1_{self.cls}'] = np.mean(greater_than_one_tracks)

            metrics[f'total_tracks_{self.cls}'] = len(self.tracks)

            for k in ['total','det_match'] + self.detection_decisions + self.tracking_decisions:
                if k == 'total':
                    metrics[f'acc_{k}_{self.cls}'] = self.logging[k+'_correct'] / (self.logging[k+'_gt'] + 0.000000000001)
                else:
                    metrics[f'recall_{k}_{self.cls}'] = self.logging[k+'_correct'] / ( self.logging[k+'_gt'] + 0.000000000001)
                    metrics[f'precision_{k}_{self.cls}'] = self.logging[k+'_correct'] / ( self.logging[k+'_num_pred'] + 0.000000000001)

                    add = metrics[f'recall_{k}_{self.cls}'] + metrics[f'precision_{k}_{self.cls}'] + 0.000000000001
                    mul = metrics[f'recall_{k}_{self.cls}'] * metrics[f'precision_{k}_{self.cls}']
                    metrics[f'f1_{k}_{self.cls}'] = 2 * (mul/add)
                    

            total_tp_decisions = torch.cat([torch.cat(x) for k,x in self.logging.items() if k.startswith('num_TP')]).sum()
            

            for k in self.logging:

                if k.startswith('num_TP') or k.startswith('num_dec'):
                    metrics[f'%_{k[4:]}'] = torch.cat(self.logging[k]).sum() / (total_tp_decisions + 0.000000000001)

                elif not ( k.endswith('_gt') or k.endswith('_correct') or k.endswith('_num_pred')):
                    # print(self.logging[k])
                    # print([x.shape for x in self.logging[k]])

                    try:
                        metrics[k] = torch.cat(self.logging[k]).mean().item()
                    except RuntimeError:
                        print('RuntimeError',k)
                    except TypeError:
                        print('TypeError for k:{}'.format(k))
            

        return metrics

    

    def get_global_track_id_preds(self, tracks_to_output, device, output_preds=True):
        """
        Args:
            tracks_to_output (torch.Tensor): list of class-specific track
                ids corresponding to output at the current frame.
            output_preds (bool): whether to compute predictions or not.
        
        Returns:
            (torch.tensor) list of global tracking ids corresponding to the current 
                detections if output_preds is True, None otherwise.
        """
        if output_preds:
            trk_id_preds = torch.tensor([self.tracks[i].id for i in tracks_to_output],device=device)
            score_preds = torch.tensor([self.tracks[i].det_confidence[-1] for i in tracks_to_output],device=device)
            cls_preds = torch.tensor([self.tracks[i].cls[-1] for i in tracks_to_output],device=device)


            if len(tracks_to_output) == 0:
                bbox_preds = torch.tensor([],device=device)
            else:
                if self.box_output_mode.lower() == 'det':
                    bbox_preds = torch.stack([self.tracks[i].det_bboxes[-1] for i in tracks_to_output],dim=0)
                elif  self.box_output_mode.lower() == 'track':
                    bbox_preds = torch.stack([self.tracks[i].refined_bboxes[-1] for i in tracks_to_output],dim=0)
                else:
                    raise ValueError("Incorrect box_output_mode")

            return trk_id_preds, score_preds, bbox_preds, cls_preds
        else:
            return None, None, None, None



    def verify_decisions(self,decisions,tp_decisions,gt_tracks,gt_track_tte,timestep,num_dets,num_tracks):
        """Debugging method to print decision error messages"""
        try:
            assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:9] == 'pos_track']) == \
                sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']), 'track decisions issue'

            assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) == \
                sum([x.size(0) for k,x in decisions.items() if k[:3] == 'det']), "detection decisions issue"
        except AssertionError as e:
            print(e)
            print("[AssertionError]  assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) ==sum([x.size(0) for k,x in decisions.items() if k[:3] == 'det'])")
            print('or')
            print("[AssertionError]  assert sum([x.size(0) for k,x in tp_decisions.items()  if k[:9] == 'pos_track']) == sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track'])")

            print('In virtual tracker step()')
            print('      decisions:',[[k,x.size(0)] for k,x in decisions.items()],sum([x.size(0) for k,x in decisions.items()]))
            print('decisions det  : ',sum([x.size(0) for k,x in decisions.items()  if k[:3] == 'det']) )
            print('decisions track: ',sum([x.size(0) for k,x in decisions.items() if k[:5] == 'track']) )
            print('      tp_decisions:',[[k,x.size(0)] for k,x in tp_decisions.items() if k[:4] == 'pos_'],sum([x.size(0) for k,x in tp_decisions.items() if k[:4] == 'pos_']))
            print('tp_decisions det  : ',sum([x.size(0) for k,x in tp_decisions.items()  if k[:7] == 'pos_det']) )
            print('tp_decisions track: ',sum([x.size(0) for k,x in tp_decisions.items() if k[:9] == 'pos_track']) )

            for k,x in tp_decisions.items():
                if k[:9] == 'pos_track':
                    print(k,x)
                    idx = torch.tensor(self.activeTracks)[x]
                    print('trkid_to_gt_trkid',self.trkid_to_gt_trkid[idx])
                    print('trkid_to_gt_tte',self.trkid_to_gt_tte[idx])

            print('gt_trk_id',gt_tracks)
            print('gt_trk_tte',gt_track_tte)

            print('timestep:',timestep)

            print('num_det',num_dets)
            print('num_track',num_tracks)
            exit(0)


























