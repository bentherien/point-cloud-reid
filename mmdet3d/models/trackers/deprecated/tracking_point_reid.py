import torch
import itertools
import timeit

import numpy as np
import torch.nn as nn

from functools import reduce

from mmdet3d.models.trackers.pc_utils import (get_crops_per_image, interpolate_per_frame, get_input_batch)
from mmdet3d.models import TRACKERS
from mmdet3d.models import builder
from mmdet3d.datasets.utils import MatchingEval

def get_labels_to_compare(det_labels,track_labels,det_lengths,track_lengths,use_lengths,device):
    accum = []
    for x in range(0,8):
        if use_lengths:
            idx1 = torch.where(reduce(torch.logical_and,[track_labels == x, track_lengths >= 2]))[0]   
            idx2 = torch.where(reduce(torch.logical_and,[det_labels == x, det_lengths >= 2]))[0]   
        else:
            idx1 = torch.where(track_labels == x)[0]   
            idx2 = torch.where(det_labels == x)[0]
        if len(idx1) == 0 or len(idx2) ==0:
            continue

        accum.append(torch.cartesian_prod(idx1,idx2))


    if accum == []: 
        return None

    return torch.cat(accum,dim=0)



@TRACKERS.register_module()
class PointReidentifier(object):

    def __init__(self,use_pc_feats,subsample_number=128,replace_all=False):
        super().__init__()
        self.subsample_number = subsample_number
        self.point_feature_set = builder.build_tracker(dict(type='PointFeatureSet',replace_all=replace_all))
        self.use_pc_feats = use_pc_feats

    def reset(self):
        self.point_feature_set.reset()

    def __call__(self,net,tracker,num_tracks,num_dets,sweeps_list,pred_cls,bbox,device):
        pts_cost_mat, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp = None, None, None, None, None, None
        if not self.use_pc_feats or num_dets == 0:
            return pts_cost_mat, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp

        # return pts_cost_mat, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp




        # def f():
        #     centered, pts_lengths_det = interpolate_per_frame(bboxes=bbox.tensor[:,:7],
        #                                                     pts=sweeps_list[0],
        #                                                     device=device)
        #     pts_batched = get_input_batch(centered,pts_lengths_det,subsample_number=self.subsample_number,device=device)
        #     # return crops,idx


        # if bbox.tensor.size(0) != 100:
        #     bbox.tensor = torch.cat([bbox.tensor for x in range(int(np.ceil(100/bbox.tensor.size(0))))],dim=0)[:100,...]

        # print('pc size:',sweeps_list[0].shape)
        # print("bboxes:",bbox.tensor.shape)
        # t = timeit.repeat(f, number=10,repeat=5)
        # print("[timeit points] mean:{} +- {}ms".format(np.mean(t),np.std(t)))
            
        # print("shape {} type {} device {}".format(bbox.tensor.shape,type(bbox.tensor),bbox.tensor.device))
        # bbox.tensor = torch.empty((0,9),dtype=torch.float32,device=device)
        centered, pts_lengths_det = interpolate_per_frame(bboxes=bbox.tensor[:,:7],
                                                            pts=sweeps_list[0],
                                                            device=device)
        pts_batched = get_input_batch(centered,pts_lengths_det,subsample_number=self.subsample_number,device=device)
        pts_lengths_det = pts_lengths_det.squeeze(0)


        # torch.save(pts_batched,'debug_pts/det_crops_{}.pt'.format(str(device)))


        with torch.no_grad():
            xyz_det, pts_feat_det = net.pts_reid_net.backbone(pts_batched[0,...],[128, 64, 32])


        if num_tracks != 0:
            
            active_track_labels = torch.stack([tracker.tracks[idx].cls[-1] for idx in tracker.activeTracks])
            pts_lengths_track = self.point_feature_set.lengths[torch.tensor(tracker.activeTracks,device=device)]
            pts_cp = get_labels_to_compare(det_labels=pred_cls,
                                            track_labels=active_track_labels,
                                            det_lengths=pts_lengths_det,
                                            track_lengths=pts_lengths_track,
                                            use_lengths=True,
                                            device=device)
            


            pts_cost_mat = torch.zeros((num_tracks,num_dets),device=device)
            # print('pts_cp in points reidentifier')
            if pts_cp is not None:
                xyz_track, pts_feat_track = self.point_feature_set.get_features(torch.tensor(tracker.activeTracks,device=device))
                

                out = net.pts_reid_net.match_forward_inference(h1=pts_feat_track[pts_cp[:,0],...],
                                                                h2=pts_feat_det[pts_cp[:,1],...],
                                                                xyz1=xyz_track[pts_cp[:,0],...],
                                                                xyz2=xyz_det[pts_cp[:,1],...])

                
                pts_cost_mat[pts_cp[:,0],pts_cp[:,1]] = out
                torch.cuda.empty_cache()
                return pts_cost_mat, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp

        return None, xyz_det, pts_feat_det, pts_lengths_det, pts_lengths_track, pts_cp



@TRACKERS.register_module()
class ImageReidentifier(object):

    def __init__(self,use_im_feats,replace_all=False):
        super().__init__()

        self.use_im_feats = use_im_feats
        self.image_feature_set = builder.build_tracker(dict(type='ImageFeatureSet',replace_all=replace_all))


    def __call__(self,net,tracker,pred_cls,orig_imgs,camera_intrinsics,lidar2camera,boxes_3d,num_tracks,num_dets,device):
        im_cost_mat, im_feats_det, vis_det, im_feats_track, vis_track, im_cp = None, None, None, None, None, None
        if not self.use_im_feats or num_dets == 0:
            return im_cost_mat, im_feats_det, vis_det, im_feats_track, vis_track, im_cp
        

        # if boxes_3d.size(0) != 100:
        #     boxes_3d = torch.cat([boxes_3d for x in range(int(np.ceil(100/boxes_3d.size(0))))],dim=0)[:100,...]

        # def f():
        #     crops, idx = get_crops_per_image(images=orig_imgs,
        #                                     ci_list=camera_intrinsics,
        #                                     l2c_list=lidar2camera,
        #                                     boxes_3d=boxes_3d,
        #                                     device=device,
        #                                     imsize=(1600,900),
        #                                     output_size=(224,224,),)
        #     return crops,idx

        # print("bboxes:",boxes_3d.shape)
        # t = timeit.repeat(f, number=10,repeat=5)
        # print("[timeit image] mean:{} +- {}ms".format(np.mean(t),np.std(t)))

        # exit(0)

        crops, idx = get_crops_per_image(images=orig_imgs,
                                            ci_list=camera_intrinsics,
                                            l2c_list=lidar2camera,
                                            boxes_3d=boxes_3d,
                                            device=device,
                                            imsize=(1600,900),
                                            output_size=(224,224,),)


        det_crops = torch.zeros((boxes_3d.size(0),)+ crops.shape[1:],device=device)
        det_crops[idx,...] = crops


        # torch.save(det_crops,'debug_images/det_crops_{}.pt'.format(str(device)))
        # exit(0)


        with torch.no_grad():
            im_feats_det_pooled, im_feats_det = net.img_reid_net.forward_inference(det_crops)
            vis_det = net.img_reid_net.vis_head(im_feats_det_pooled).argmax(dim=1)

            # print(im_feats_det.shape)
            b,s,c = im_feats_det.shape
            im_feats_det = net.img_reid_net.downsample(im_feats_det.reshape(-1,c)).reshape(b,net.img_reid_net.downsample_dim,s)
            


        if num_tracks != 0:
            
            active_track_labels = torch.stack([tracker.tracks[idx].cls[-1] for idx in tracker.activeTracks])
            pts_cp = get_labels_to_compare(det_labels=pred_cls,
                                           track_labels=active_track_labels,
                                           det_lengths=None,
                                           track_lengths=None,
                                           use_lengths=False,
                                           device=device)


            im_cost_mat = torch.zeros((num_tracks,num_dets),device=device)
            if pts_cp is not None:
                im_feat_track = self.image_feature_set.get_features(torch.tensor(tracker.activeTracks,device=device))
                out = net.img_reid_net.match_forward_inference(h1=im_feat_track[pts_cp[:,0],...],
                                                               h2=im_feats_det[pts_cp[:,1],...],)

                im_cost_mat[pts_cp[:,0],pts_cp[:,1]] = out
                torch.cuda.empty_cache()
                return im_cost_mat, im_feats_det, vis_det, im_feats_track, vis_track, im_cp



        return None, im_feats_det, vis_det, im_feats_track, vis_track, im_cp
        



@TRACKERS.register_module()
class MatchingMetrics(object):
    def __init__(self):
        super().__init__()
        self.matching_eval = MatchingEval()


    def log_match_preds(self,tracker,tp_decisions,decisions,pts_cost_mat,im_cost_mat,pts_lengths_track,pts_lengths_det,device):
        mm_preds = torch.zeros_like(pts_cost_mat,dtype=torch.float32,device=device)
        mm_preds[decisions['track_match'],decisions['det_match']] = 1

        targets = torch.cat([torch.ones(len(tp_decisions['pos_track_match']),device=device),
                             torch.zeros(len(tp_decisions['neg_track_match']),device=device)],dim=0)
        
        preds = dict(motion_model=torch.cat([mm_preds[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']],
                                             mm_preds[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']]],dim=0),
                     pts_reid=torch.cat([pts_cost_mat[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']],
                                         pts_cost_mat[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']]],dim=0),
                     img_reid=torch.cat([im_cost_mat[tp_decisions['pos_track_match'],tp_decisions['pos_det_match']],
                                         im_cost_mat[tp_decisions['neg_track_match'],tp_decisions['neg_det_match']]],dim=0),
                    )
        

        pc_pairwise_lengths = torch.cat([torch.cat([pts_lengths_track[tp_decisions['pos_track_match']],pts_lengths_track[tp_decisions['neg_track_match']]],dim=0).unsqueeze(1),
                                         torch.cat([pts_lengths_det[tp_decisions['pos_det_match']],pts_lengths_det[tp_decisions['neg_det_match']]],dim=0).unsqueeze(1)],dim=1)

        tracker.log_update(dict(matching_metrics = (targets,preds,pc_pairwise_lengths,)))


    def merge_match_metrics(self,logging):
        matching_metrics = logging['matching_metrics']
        # print('matching_metrics',matching_metrics)
        preds = dict(motion_model = torch.cat([x[1]['motion_model'].clone().cpu() for x in matching_metrics],dim=0),
                     pts_reid = torch.cat([x[1]['pts_reid'].clone().cpu() for x in matching_metrics],dim=0),
                     img_reid = torch.cat([x[1]['img_reid'].clone().cpu() for x in matching_metrics],dim=0))

        targets = torch.cat([x[0].clone().cpu() for x in matching_metrics],dim=0)

        pc_pairwise_lengths = torch.cat([x[2].clone().cpu() for x in matching_metrics],dim=0)

        return dict(matching_metrics = (targets,preds,pc_pairwise_lengths,))



    def eval_match(self, logging, display=True):
        targets,preds,pc_pairwise_lengths = self.merge_match_metrics(logging)['matching_metrics']

        motion_model_preds = (nn.Sigmoid()(preds['motion_model']) > 0.5).float()
        filter_ = torch.where(motion_model_preds != targets)

        all_log_vars = {}
        log_vars = {}
        for k,v in preds.items():
            temp = {}
            preds = (nn.Sigmoid()(v) > 0.5).float()
            f1_precision_recall = self.matching_eval.f1_precision_recall(preds, targets)
            f1_precision_recall['accuracy'] = (preds == targets).float().mean().item()
            f1_precision_recall['accuracy_MME'] = (preds[filter_] == targets[filter_]).float().mean().item()

            temp.update(f1_precision_recall)
            temp.update(self.matching_eval.evaluate_points(v, targets, pc_pairwise_lengths))
            all_log_vars[k] = temp
            log_vars[k] = f1_precision_recall

            if display:

                print("=============================================")
                print("for key: {} \n[all] Num Pos:{} Num Neg.:{}\n[MME] Num Pos:{} Num Neg.:{}".format(
                    k,targets.sum().item(),(1-targets).sum().item(),targets[filter_].sum().item(),(1-targets[filter_]).sum().item()))
                print("=============================================")
                for kk,vv in f1_precision_recall.items():
                    print("{}: {}".format(kk,vv))

        return log_vars, all_log_vars

