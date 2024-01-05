import torch.nn as nn
from mmdet3d.models import FUSIONMODELS
from transformers import (AutoImageProcessor, 
                          AutoConfig, 
                          AutoModel, 
                          BeitModel, 
                          AutoFeatureExtractor, 
                          ViTForImageClassification, 
                          DeiTForImageClassificationWithTeacher)
import torch.nn.functional as F

# from torch.nn.modules.transformer import (TransformerEncoder, TransformerDecoder, 
#                                           TransformerDecoderLayer, TransformerEncoderLayer,)
# from .vn_dgcnn_cls import VNDGCNN
# from torch_geometric.nn.conv import GATv2Conv,GatedGraphConv
# from torch_geometric.data import Data
# from .node_pooling import GatedPooling
# from .loftr import LocalFeatureTransformer


from .lanegcn_nets import PostRes,LinearRes
from .backbone_net import Pointnet_Backbone
from .dgcnn_orig import DGCNN

from mmdet.models import BaseDetector
import torch.distributed as dist
from pytorch3d.loss import chamfer_distance

import torch
import copy
import time 

from .pointnet import PointNet

from .attention import corss_attention, local_self_attention, cross_lin_attn




module_obj = {
    'Linear':nn.Linear,

    'ReLU':nn.ReLU,
    'LSTM':nn.LSTM,
    'GroupNorm':nn.GroupNorm,
    'Embedding':nn.Embedding,

    # 'MultiheadAttention':nn.MultiheadAttention,
    # 'TransformerEncoder':TransformerEncoder,
    # 'TransformerEncoderLayer':TransformerEncoderLayer,
    # 'TransformerDecoder':TransformerDecoder,
    # 'TransformerDecoderLayer':TransformerDecoderLayer,
    # 'vn_dgcnn':VNDGCNN,

    # 'LocalFeatureTransformer':LocalFeatureTransformer,
    # 'GatedPooling':GatedPooling,
    # 'GATv2Conv':GATv2Conv,
    # 'GatedGraphConv':GatedGraphConv,

    'LayerNorm':nn.LayerNorm,
    'PostRes':PostRes,
    'LinearRes':LinearRes,

    'Pointnet_Backbone':Pointnet_Backbone,
    'corss_attention':corss_attention,
    'local_self_attention':local_self_attention,

    'Conv1d':nn.Conv1d,
    'Conv2d':nn.Conv2d,
    'BatchNorm1d':nn.BatchNorm1d,
    'Sigmoid':nn.Sigmoid,
    'cross_lin_attn':cross_lin_attn,
    'dgcnn':DGCNN,
    'PointNet':PointNet
}

def build_module(cfg):
    if cfg == None or cfg == {}:
        return None

    if isinstance(cfg, list):
        return build_sequential(cfg)

    cls_ = module_obj[cfg['type']]
    del cfg['type']
    return cls_(**cfg)


def build_sequential(module_list):
    if module_list == None or module_list == {}:
        return None
        
    modules = []
    for cfg in module_list:
        modules.append(build_module(cfg))
    return nn.Sequential(*modules)

def build_decisions(decisions):
    if decisions == None or decisions == {}:
        return None

    emb = decisions['embedding']
    return nn.ModuleDict({k:build_module(copy.deepcopy(emb)) for k in decisions if k != 'embedding'})

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


@FUSIONMODELS.register_module()
class ReIDNet(BaseDetector):
    def __init__(self,hidden_size,backbone,cls_head,match_head,shape_head,fp_head,downsample,
                 cross_stage1,local_stage1,cross_stage2,local_stage2,match_type='xcorr', pool_type='max',combine='cat',
                 compute_summary=True,train_cfg=None,test_cfg=None, backbone_list=[512,256,128],use_dgcnn=False,
                 losses_to_use=dict(kl=True,match=True,cls=True,shape=True,fp=True,dense=False,),output_sequence_size=32,
                 alpha=dict(kl=1,match=1,cls=1,shape=1,fp=1,triplet=1,dense=1),triplet_sample_num=5,
                 triplet_loss=dict(margin=0.2,p=2),eval_only=False,use_o=False,eval_flip=False):
        super().__init__()
        print('Entering ReIDNet')
        
        self.eval_only = eval_only
        self.hidden_size = hidden_size
        self.match_type = match_type
        self.backbone = build_module(backbone)
        self.cls_head = build_module(cls_head)
        self.match_head = build_module(match_head)
        self.shape_head = build_module(shape_head)
        self.fp_head = build_module(fp_head)
        self.downsample = build_module(downsample)
        
        self.cross_stage1 = build_module(cross_stage1)                 
        self.local_stage1 = build_module(local_stage1)

        self.cross_stage2 = build_module(cross_stage2)          
        self.local_stage2 = build_module(local_stage2)

        self.losses_to_use = dict(kl=False,match=True,cls=False,shape=False,fp=False,dense=False,)
        self.losses_to_use.update(losses_to_use)

        self.backbone_list = backbone_list
        self.output_sequence_size = output_sequence_size
        self.pool_type = pool_type

        self.maxpool = nn.MaxPool1d(self.output_sequence_size)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(log_target=True,reduction='none')
        self.lsmx = nn.LogSoftmax(dim=1)
        self.smooth_l1 = nn.SmoothL1Loss(reduce=True, reduction='mean', beta=1.0)
        
        self.triplet_sample_num = triplet_sample_num
        self.triplet_loss = nn.TripletMarginLoss(**triplet_loss)
        self.alpha = alpha
        self.use_o = use_o
        self.eval_flip = eval_flip


        self.verbose = False

        self.sampling = None

        self.compute_summary = compute_summary
        print('Exiting ReIDNet')

        self.use_dgcnn = use_dgcnn
        self.combine = combine



    def tracking_train(self):

        for param in self.parameters():
            param.requires_grad = False
        for param in self.cross_stage1.parameters():
            param.requires_grad = True
        for param in self.cross_stage2.parameters():
            param.requires_grad = True
        for param in self.match_head.parameters():
            param.requires_grad = True
        

    def tracking_eval(self):
        for param in self.parameters():
            param.requires_grad = False



    def forward_inference(self,pts_batched):
        with torch.no_grad():
            return self.backbone(pts_batched,self.backbone_list)
            

    

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # print(data_batch)
        # exit(0)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def accum_val_log(self):
        match_gt = torch.cat(self.val_log['val_match_gt'],dim=0)
        match_preds = torch.cat(self.val_log['val_match_preds'],dim=0)
        match_acc = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match_gt).float().mean().item()

        cls_gt = torch.cat(self.val_log['val_cls_gt'],dim=0)
        cls_preds = torch.cat(self.val_log['val_cls_preds'],dim=0)
        cls_acc = cls_preds.argmax(dim=1).eq(torch.cat(cls_gt,dim=0)).float().mean().item()

        del self.val_log['val_match_gt']
        del self.val_log['val_match_preds']
        del self.val_log['val_cls_gt']
        del self.val_log['val_cls_preds']
        self.val_log.update({'val_match_acc':match_acc,'val_cls_acc':cls_acc})
        return self.val_log
    
    
    def xcorr_eff(self, o1, xyz1, o2, xyz2,combine='add'):
        o1__ = self.cross_stage1(o1, xyz1, o2, xyz2)
        o2__ = self.cross_stage1(o2, xyz2, o1, xyz1)
        
        o1 = self.cross_stage2(o1__, xyz1, o2__, xyz2)
        o2 = self.cross_stage2(o2__, xyz2, o1__, xyz1)
        
        if self.combine == 'add':
            out = o1 + o2
        elif self.combine == 'minus':
            out = o1 - o2
        elif self.combine == 'cat':
            out = torch.cat([o1,o2],dim=1)
        elif self.combine == 'point-cat':
            out = torch.cat([o1,o2],dim=2)
        
        return out, o1, o2


    def xcorr(self, search_feat, search_xyz, template_feat, template_xyz):       
        search_feat1_a = self.cross_stage1(search_feat, search_xyz, template_feat, template_xyz)
        search_feat1_b = self.local_stage1(search_feat1_a, search_xyz)
        search_feat2_a = self.cross_stage2(search_feat1_b, search_xyz, template_feat, template_xyz)
        search_feat2_b = self.local_stage2(search_feat2_a, search_xyz)

        return search_feat2_b

    def xcorr_baseline(self, search_feat, search_xyz, template_feat, template_xyz):       
        search_feat1_a = self.cross_stage1(search_feat, search_xyz, template_feat, template_xyz)
        # search_feat1_b = self.local_stage1(search_feat1_a, search_xyz)
        search_feat2_a = self.cross_stage2(search_feat1_a, search_xyz, template_feat, template_xyz)
        # search_feat2_b = self.local_stage2(search_feat2_a, search_xyz)

        return search_feat2_a

    def preprocess_inputs(self, sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        dense_1 = torch.stack(dense_1,dim=0)
        dense_2 = torch.stack(dense_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)

        if self.eval_flip:
            return sparse_2,sparse_1,dense_2,dense_1,label_2,label_1,id_2,id_1
        else:
            return sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2

    def preprocess_inputs_size(self, sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        dense_1 = torch.stack(dense_1,dim=0)
        dense_2 = torch.stack(dense_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)
        size_1 = torch.cat(size_1,dim=0)
        size_2 = torch.cat(size_2,dim=0)
        
        return sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2

    def preprocess_inputs_size_vis(self, sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        dense_1 = torch.stack(dense_1,dim=0)
        dense_2 = torch.stack(dense_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)
        size_1 = torch.cat(size_1,dim=0)
        size_2 = torch.cat(size_2,dim=0)
        vis_1 = torch.cat(vis_1,dim=0)
        vis_2 = torch.cat(vis_2,dim=0)
        
        return sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2

    def siamese_forward(self,sparse_1,sparse_2):
        assert sparse_1.shape == sparse_2.shape
        b,num_points,_ = sparse_1.shape
        # print('sparse_1',sparse_1.shape)
        # print(torch.cat([sparse_1,sparse_2],dim=0).permute(0,2,1).shape)
        if self.use_dgcnn:
            xyz, h = self.backbone(torch.cat([sparse_1,sparse_2],dim=0).permute(0,2,1),self.backbone_list)
            # print('before',h.shape)
            h = h.permute(0,2,1)
            h = h.reshape(-1,h.shape[-1])
            # print(h.shape)
            h = self.downsample(h).reshape(2*b,num_points,-1).permute(0,2,1)
            # print('after',xyz.shape,h.shape)
            return xyz[:b,...].permute(0,2,1), xyz[b:,...].permute(0,2,1), h[:b,...], h[b:,...]
        elif type(self.backbone) == PointNet:
            xyz, h = self.backbone(torch.cat([sparse_1,sparse_2],dim=0).permute(0,2,1),self.backbone_list)
            print(xyz.shape,h.shape)
            return xyz[:b,...].permute(0,2,1), xyz[b:,...].permute(0,2,1), h[:b,...], h[b:,...]
        else:
            xyz, h = self.backbone(torch.cat([sparse_1,sparse_2],dim=0),self.backbone_list)
            # print(xyz.shape,h.shape)
            return xyz[:b,...], xyz[b:,...], h[:b,...], h[b:,...]





    def get_match_supervision(self,h1,h2,xyz1,xyz2,id_1,id_2):
        if self.sampling == 'cartesian':
            match_cp = torch.cartesian_prod(torch.arange(batch),torch.arange(batch))
            h1_cp, xyz1 = h1[match_cp[:,0]], xyz1[match_cp[:,0]]
            h2_cp, xyz2 = h2[match_cp[:,1]], xyz2[match_cp[:,1]]
            match = ( id_1[match_cp[:,0]] == id_2[match_cp[:,1]] ).float()
            return h1_cp, h2_cp, xyz1, xyz2, match
        else:
            return h1, h2, xyz1, xyz2, ( id_1 == id_2 ).float()

    def cls_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['cls']:
            # input_ = self.maxpool(h.permute(0,2,1)).squeeze(-1)
            input_ = self.get_pooled_feats(h)
            cls_preds = self.cls_head(input_).squeeze(1)
            cls_loss = self.ce(cls_preds,target)


            cls_loss = cls_loss * self.alpha['cls']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'cls_loss'] = cls_loss.item()
                log_vars[prefix+'cls_acc'] = cls_preds.argmax(dim=1).eq(target).float().mean().item()
        else:
            cls_preds = None
            cls_loss = torch.tensor(0.,requires_grad=True,device=device)

        return cls_preds, cls_loss


    def fp_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['fp']:
            # input_ = self.maxpool(h.permute(0,2,1)).squeeze(-1)
            input_ = self.get_pooled_feats(h)
            fp_preds = self.fp_head(input_).squeeze(1)
            target = ( target > 9 ).float()
            fp_loss = self.bce(fp_preds, target)
            fp_loss = fp_loss * self.alpha['fp']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'fp_loss'] = fp_loss.item()
                log_vars[prefix+'fp_acc'] = (nn.Sigmoid()(fp_preds) > 0.5).float().eq(target).float().mean().item()
        else:
            fp_preds = None
            fp_loss = torch.tensor(0.,requires_grad=True,device=device)

        return fp_preds, fp_loss


    def match_forward(self,h1,h2,xyz1,xyz2,match,log_vars,device,prefix=''):
        o1, o2 = None, None
        if self.losses_to_use['match']:
            if self.match_type == 'xcorr':

                match_in = self.xcorr(h1,xyz1,h2,xyz2)
                # match_in = self.maxpool(match_in.permute(0,2,1)
                match_in = self.get_pooled_feats(match_in)
                match_preds = self.match_head(match_in).squeeze(1)
                match_loss = self.bce(match_preds,match)

            elif self.match_type == 'xcorr-baseline':

                match_in = self.xcorr_baseline(h1,xyz1,h2,xyz2)
                # match_in = self.maxpool(match_in.permute(0,2,1)
                # print("before pooling",match_in.shape)
                match_in = self.get_pooled_feats(match_in)
                # print("after pooling",match_in.shape)
                match_preds = self.match_head(match_in).squeeze(1)
                match_loss = self.bce(match_preds,match)

            elif self.match_type == 'xcorr_eff':

                match_in, o1, o2 = self.xcorr_eff(h1,xyz1,h2,xyz2,self.combine)
                match_in = self.get_pooled_feats(match_in)
                match_preds = self.match_head(match_in).squeeze(1)
                match_loss = self.bce(match_preds,match)

            elif self.match_type == 'concat':
                cat = torch.cat([self.get_pooled_feats(h1),
                                 self.get_pooled_feats(h2)],dim=1)
                match_preds = self.match_head(cat).squeeze(1)
                match_loss = self.bce(match_preds,match)
            else:
                raise NotImplementedError

            match_loss = match_loss * self.alpha['match']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'match_loss'] = match_loss.item()
                log_vars[prefix+'match_acc'] = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match).float().mean().item()

                gt_bins = torch.bincount(match.long())
                log_vars[prefix+'num_preds_0'] = gt_bins[0].item()
                log_vars[prefix+'num_preds_1'] = gt_bins[1].item() if len(gt_bins) > 1 else 0

                pred_bins = torch.bincount((nn.Sigmoid()(match_preds) > 0.5).long())
                log_vars[prefix+'num_gt_0'] = pred_bins[0].item()
                log_vars[prefix+'num_gt_1'] = pred_bins[1].item() if len(pred_bins) > 1 else 0
        else:
            match_preds = None
            match_loss = torch.tensor(0.,requires_grad=True,device=device)

        return match_preds, match_loss, (o1, o2)



    def match_forward_inference(self,h1,h2,xyz1,xyz2):
        if self.match_type == 'xcorr':
            match_in = self.xcorr(h1,xyz1,h2,xyz2)
            match_in = self.get_pooled_feats(match_in)
            match_preds = self.match_head(match_in).squeeze(1)

        elif self.match_type == 'xcorr_eff':
                match_in, o1, o2 = self.xcorr_eff(h1,xyz1,h2,xyz2,self.combine)
                match_in = self.get_pooled_feats(match_in)
                match_preds = self.match_head(match_in).squeeze(1)

        elif self.match_type == 'concat':
            cat = torch.cat([self.maxpool(h1.permute(0,2,1)).squeeze(-1),
                                self.maxpool(h2.permute(0,2,1)).squeeze(-1)],dim=1)
            match_preds = self.match_head(cat).squeeze(1)
        else:
            raise NotImplementedError

        return match_preds




    def get_kl_loss(self,h1,h2,match,log_vars,device,prefix=''):

        if self.losses_to_use['kl']:
            kl_loss = self.kl(self.lsmx(h1.reshape(h1.size(0),-1)),self.lsmx(h2.reshape(h2.size(0),-1))).mean(dim=1)
            where_no_match = torch.where(match == 0)
            kl_loss[where_no_match] = kl_loss[where_no_match] * -1
            kl_loss = kl_loss[match==0].mean() + kl_loss[match==1].mean()

            kl_loss = kl_loss * self.alpha['kl']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'kl_loss'] = kl_loss.item()
        else:
            kl_loss = torch.tensor(0.,requires_grad=True,device=device)

        return kl_loss



    def get_dense_loss(self,h,d1,d2,ids,log_vars,device,prefix=''):
        if self.losses_to_use['dense']:
            fp_filter = torch.where(ids != -1)[0]
            with torch.no_grad():
                _, _, d1, d2 = self.siamese_forward(d1,d2)
            d = torch.cat([d1,d2],dim=0)
            # print(d.shape)
            # print(h.shape)
            # print(h.reshape(h.size(0),-1).shape)
            dense_loss = self.smooth_l1(h[fp_filter].reshape(h.size(0),-1),d[fp_filter].reshape(d.size(0),-1))

            dense_loss = dense_loss * self.alpha['dense']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'dense_loss'] = dense_loss.item()
        else:
            dense_loss = torch.tensor(0.,requires_grad=True,device=device)

        return dense_loss



    def shape_forward(self,h,target,log_vars,device,prefix=''):

        if self.losses_to_use['shape']:
            shape_preds = self.shape_head(h.permute(0,2,1))
            # shape_preds = self.shape_head(self.get_pooled_feats(h)).reshape(-1,2048,3,)
            shape_loss,_ = chamfer_distance(shape_preds,target)
            shape_loss = shape_loss * self.alpha['shape']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'shape_loss'] = shape_loss.item()

        else:
            shape_preds = None
            shape_loss = torch.tensor(0.,requires_grad=True,device=device)

        return shape_preds, shape_loss


    def get_pooled_feats(self,h_cat):
        if self.pool_type == 'max':
            return self.maxpool(h_cat.permute(0,2,1)).squeeze(-1)
        elif self.pool_type == 'both':
            x1 = F.adaptive_max_pool1d(h_cat, 1).view(h_cat.size(0), -1)
            x2 = F.adaptive_avg_pool1d(h_cat, 1).view(h_cat.size(0), -1)
            return torch.cat((x1, x2), 1)
        else:
            raise NotImplementedError



    def get_triplet_loss(self,h1,h2,id1,id2,match,log_vars,device,prefix=''):

        if self.losses_to_use['triplet']:
            match_idx = torch.where(match == 1)[0]
            matches = id1[match_idx]

            h_cat = torch.cat([h1,h2],dim=0)
            id_ = torch.cat([id1,id2],dim=0)

            a, p, n = [], [], []
            for i in range(matches.size(0)):


                m,idx = matches[i],match_idx[i]

                sample_pool = torch.where(id_ != m)[0]
                neg_idx_to_use = torch.multinomial(torch.ones(sample_pool.size(0)), self.triplet_sample_num, replacement=(len(sample_pool) < self.triplet_sample_num)).to(device)
                neg_idx_to_use = sample_pool[neg_idx_to_use]

                a.append(torch.full((self.triplet_sample_num,),idx,device=device))
                p.append(torch.full((self.triplet_sample_num,),idx,device=device))
                n.append(neg_idx_to_use)

            a = torch.cat(a,dim=0)
            p = torch.cat(p,dim=0)
            n = torch.cat(n,dim=0)

            a = h1.reshape(h1.size(0),-1)[a,...]
            p = h2.reshape(h1.size(0),-1)[p,...]
            n = h_cat.reshape(h_cat.size(0),-1)[n,...]


            triplet_loss = self.triplet_loss(anchor=a,
                                             positive=p,
                                             negative=n)
            triplet_loss = triplet_loss * self.alpha['triplet']


            if self.compute_summary and log_vars != None:
                log_vars[prefix+'triplet_loss'] = triplet_loss.item()
        else:
            triplet_loss = torch.tensor(0.,requires_grad=True,device=device)

        
        return triplet_loss



    def forward_train(self,sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2):
        if self.eval_only:
            exit(0)

        
        log_vars = {}
        losses = {}

        sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2 = self.preprocess_inputs(sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2)
        device = sparse_1.device
        # print(sparse_1.shape,sparse_2.shape,dense_1.shape,dense_2.shape,label_1.shape,label_2.shape,id_1.shape,id_2.shape)
        # print(sparse_1.shape)
        # print(sparse_1.shape) 
        #siamese_forward
        xyz1, xyz2, h1, h2 = self.siamese_forward(sparse_1,sparse_2)
        h_cat = torch.cat([h1,h2],dim=0)

        fp_filter = torch.where(torch.cat([id_1,id_2],dim=0) != -1)[0]
        # print(xyz1.shape, xyz2.shape, h1.shape, h2.shape)

        #cls forward
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        #FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        #shape forward
        shape_preds, shape_loss = self.shape_forward(h_cat[fp_filter],torch.cat([dense_1,dense_2],dim=0)[fp_filter],log_vars,device,prefix='')


        h1, h2, xyz1, xyz2, match = self.get_match_supervision(h1,h2,xyz1,xyz2,id_1,id_2)
        #match forward
        match_preds, match_loss, (o1, o2) = self.match_forward(h1,h2,xyz1,xyz2,match,log_vars,device,prefix='')
        #kl forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')

        dense_loss = self.get_dense_loss(h_cat,dense_1,dense_2,torch.cat([id_1,id_2],dim=0),log_vars,device,prefix='')


        #triplet forward
        if self.use_o:
            h1, h2 = self.get_pooled_feats(o1), self.get_pooled_feats(o2)
        
        
        triplet_loss = self.get_triplet_loss(h1,h2,id_1,id_2,match,log_vars,device,prefix='')

        losses['reid_loss'] = match_loss + shape_loss + cls_loss + kl_loss + fp_loss + triplet_loss + dense_loss

        return losses, log_vars


    def forward_test(self,sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2,*args,**kwargs):
        results = {}
        log_vars = None

        sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2 = \
            self.preprocess_inputs_size_vis(sparse_1,sparse_2,dense_1,dense_2,label_1,label_2,id_1,id_2,size_1,size_2,vis_1,vis_2)
        device = sparse_1.device

        fp_filter = torch.where(torch.cat([id_1,id_2],dim=0) != -1)[0]
        # print(sparse_1.shape,sparse_2.shape,dense_1.shape,dense_2.shape,label_1.shape,label_2.shape,id_1.shape,id_2.shape)

        #siamese_forward
        xyz1, xyz2, h1, h2 = self.siamese_forward(sparse_1,sparse_2)
        h_cat = torch.cat([h1,h2],dim=0)

        #cls forward
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        #FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        #shape forward
        shape_preds, shape_loss = self.shape_forward(h_cat[fp_filter],torch.cat([dense_1,dense_2],dim=0)[fp_filter],log_vars,device,prefix='')

        dense_loss = self.get_dense_loss(h_cat,dense_1,dense_2,torch.cat([id_1,id_2],dim=0),log_vars,device,prefix='')


        h1, h2, xyz1, xyz2, match = self.get_match_supervision(h1,h2,xyz1,xyz2,id_1,id_2)
        #match forward
        match_preds, match_loss, (o1,o2) = self.match_forward(h1,h2,xyz1,xyz2,match,log_vars,device,prefix='')
        #kl forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')

        results['val_dense_loss'] = torch.tensor([dense_loss])
        results['val_fp_loss'] = torch.tensor([fp_loss])
        results['val_match_loss'] = torch.tensor([match_loss])
        results['val_shape_loss'] = torch.tensor([shape_loss])
        results['val_cls_loss'] = torch.tensor([cls_loss])
        results['val_kl_loss'] = torch.tensor([kl_loss])
        results['val_match_preds'] = match_preds
        results['val_match_gt'] = match
        results['val_cls_preds'] = cls_preds
        results['val_cls_gt'] = torch.cat([label_1,label_2],dim=0)
        results['val_fp_preds'] = fp_preds
        results['val_fp_gt'] = ( torch.cat([label_1,label_2],dim=0) > 9 ).float()
        results['match_classes'] = torch.cat([label_1.unsqueeze(1),label_2.unsqueeze(1)],dim=1)

        results['is_fp'] = torch.logical_or( (label_1 > 9) , (label_2 > 9) )
        results['num_points'] = torch.cat([size_1.unsqueeze(1),size_2.unsqueeze(1)],dim=1)
        results['val_vis_gt_all'] = torch.cat([vis_1.unsqueeze(1),vis_2.unsqueeze(1)],dim=1)


        return [ results ]


    #MMDET3d API

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        
        if self.verbose:
            t1 = time.time()
            print("{} Starting train_step()".format(self.log_msg()))


        losses, log_vars_train = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars.update(log_vars_train)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))
        
        if self.verbose:
            print("{} Ending train_step() after {}s".format(self.log_msg(),time.time()-t1))

        return outputs




    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))

        return outputs


    # def train_step(self, data_batch, optimizer,*args, **kwargs):
    #     return self.forward(data_batch,*args, **kwargs)


    def extract_feat(self,*args,**kwargs):
        raise NotImplementedError

    def show_result(self):
        raise NotImplementedError

    def aug_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def simple_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def init_weights(self):
        pass


@FUSIONMODELS.register_module()
class ReIDNetCosine(ReIDNet):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def match_forward(self,h1,h2,xyz1,xyz2,match,log_vars,device,prefix=''):
        if self.losses_to_use['match']:
            h1_ = self.get_pooled_feats(h1)
            h2_ = self.get_pooled_feats(h2)

            match_preds = F.cosine_similarity(h1_, h2_, dim=1) * 10 #scale to make it easier to learn
            match_loss = self.bce(match_preds)
            match_loss = match_loss * self.alpha['match']

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'match_loss'] = match_loss.item()
                log_vars[prefix+'match_acc'] = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match).float().mean().item()

                gt_bins = torch.bincount(match.long())
                log_vars[prefix+'num_preds_0'] = gt_bins[0].item()
                log_vars[prefix+'num_preds_1'] = gt_bins[1].item() if len(gt_bins) > 1 else 0

                pred_bins = torch.bincount((nn.Sigmoid()(match_preds) > 0.5).long())
                log_vars[prefix+'num_gt_0'] = pred_bins[0].item()
                log_vars[prefix+'num_gt_1'] = pred_bins[1].item() if len(pred_bins) > 1 else 0
        else:
            match_preds = None
            match_loss = torch.tensor(0.,requires_grad=True,device=device)

        return match_preds, match_loss

def get_image_model(model):
    if model == 'beit':
        image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        backbone = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    elif model == 'deit-tiny':
        image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
        backbone = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-tiny-distilled-patch16-224',output_hidden_states=True)
    elif model == 'deit-tiny-no-pt':
        image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
        config = AutoConfig.from_pretrained("facebook/deit-tiny-patch16-224",output_hidden_states=True)
        backbone = AutoModel.from_config(config,)
    elif model == 'deit-base-no-pt':
        image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        config = AutoConfig.from_pretrained("facebook/deit-base-patch16-224",output_hidden_states=True)
        backbone = AutoModel.from_config(config,)
    elif model == 'deit-small':
        image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-distilled-patch16-224')
        backbone = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-small-distilled-patch16-224',output_hidden_states=True)
    elif model == 'deit-base':
        image_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        backbone = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-224',output_hidden_states=True)
    else:
        raise NotImplementedError("Not implemented for model: {}".format(model))
    return image_processor, backbone



@FUSIONMODELS.register_module()
class ImageReIDNet(BaseDetector):
    def __init__(self,backbone,cls_head,match_head,vis_head,fp_head,downsample,
                 cross_lin_attn,combine='cat',dim=768,downsample_dim=128,
                 losses_to_use=dict(kl=False,match=True,cls=True,shape=True,fp=True,triplet=True,),
                 alpha=dict(kl=1,match=1,cls=1,shape=1,fp=1,triplet=1,vis=1),
                 pool_type='both',compute_summary=True, output_sequence_size=198,
                 train_cfg=None,test_cfg=None,freeze_backbone=False,triplet_sample_num=5,
                 match_type='xcorr_eff',triplet_loss=dict(margin=0.2,p=2),eval_only=False):
        super().__init__()
        
        self.eval_only = eval_only
        self.backbone_name = backbone
        self.image_processor, self.backbone = get_image_model(backbone)
        self.cross_stage1 = build_module(copy.deepcopy(cross_lin_attn))
        self.cross_stage2 = build_module(copy.deepcopy(cross_lin_attn))
        self.cls_head = build_module(cls_head)
        self.match_head = build_module(match_head)
        self.vis_head = build_module(vis_head)
        self.fp_head = build_module(fp_head)
        self.downsample = build_module(downsample)
        
        self.combine = combine
        self.dim = dim
        self.downsample_dim = downsample_dim
        self.losses_to_use = losses_to_use
        self.compute_summary = compute_summary
        self.pool_type = pool_type
        self.maxpool = nn.MaxPool1d(output_sequence_size)
        self.match_type = match_type
        
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(log_target=True,reduction='none')
        self.lsmx = nn.LogSoftmax(dim=1)

        self.verbose = False

        self.sampling = None
        
        self.triplet_sample_num = triplet_sample_num
        self.triplet_loss = nn.TripletMarginLoss(**triplet_loss)
        self.alpha = alpha


        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        

    def tracking_train(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def tracking_eval(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def xcorr_eff(self, o1, o2,combine='add'):
        o1__ = self.cross_stage1(o1, o2)
        o2__ = self.cross_stage1(o2, o1)

        o1 = self.cross_stage2(o1__, o2__)
        o2 = self.cross_stage2(o2__, o1__)

        if self.combine == 'add':
            out = o1 + o2
        elif self.combine == 'minus':
            out = o1 - o2
        elif self.combine == 'cat':
            out = torch.cat([o1,o2],dim=1)
        elif self.combine == 'point-cat':
            out = torch.cat([o1,o2],dim=2)

        return out
        
    def forward_inference(self,images):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=images)
            if 'deit' in self.backbone_name:
                outputs = outputs.hidden_states[-1]
            elif self.backbone_name == 'beit':
                outputs = outputs.last_hidden_state
            else:
                raise NotImplementedError("Not implemented for model: {}".format(self.backbone_name))


            
            return self.get_pooled_feats(outputs.permute(0,2,1)), outputs
    
    
    def siamese_forward(self,sparse_1,sparse_2):
        assert sparse_1.shape == sparse_2.shape
        b = sparse_1.size(0)

        outputs = self.backbone(pixel_values=torch.cat([sparse_1,sparse_2],dim=0))
        if  'deit' in self.backbone_name:
            outputs = outputs.hidden_states[-1]
        elif self.backbone_name == 'beit':
            outputs = outputs.last_hidden_state
        else:
            raise NotImplementedError("Not implemented for model: {}".format(self.backbone_name))
            
        return outputs[:b,...].permute(0,2,1), outputs[b:,...].permute(0,2,1)

    def get_match_supervision(self,h1,h2,id_1,id_2):
        if self.sampling == 'cartesian':
            match_cp = torch.cartesian_prod(torch.arange(batch),torch.arange(batch))
            h1_cp = h1[match_cp[:,0]]
            h2_cp = h2[match_cp[:,1]]
            match = ( id_1[match_cp[:,0]] == id_2[match_cp[:,1]] ).float()
            return h1_cp, h2_cp, match
        else:
            return h1, h2, ( id_1 == id_2 ).float()

    def cls_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['cls']:
            # input_ = self.maxpool(h.permute(0,2,1)).squeeze(-1)
            input_ = self.get_pooled_feats(h)
            cls_preds = self.cls_head(input_).squeeze(1)
            cls_loss = self.ce(cls_preds,target)

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'cls_loss'] = cls_loss.item()
                log_vars[prefix+'cls_acc'] = cls_preds.argmax(dim=1).eq(target).float().mean().item()
        else:
            cls_preds = None
            cls_loss = torch.tensor(0.,requires_grad=True,device=device)


        cls_loss = cls_loss * self.alpha['cls']
        return cls_preds, cls_loss

    
    def vis_forward(self,h,target,ids,log_vars,device,prefix=''):
        if self.losses_to_use['vis']:
            fp_filter = torch.where(torch.logical_and(ids != -1,target != -1))
            h = h[fp_filter]
            target = target[fp_filter]#.squeeze(1)
            # input_ = self.maxpool(h.permute(0,2,1)).squeeze(-1)
            input_ = self.get_pooled_feats(h)
            vis_preds = self.vis_head(input_).squeeze(1)
            # print(vis_preds.shape,target.shape)
            vis_loss = self.ce(vis_preds,target)

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'vis_loss'] = vis_loss.item()
                log_vars[prefix+'vis_acc'] = vis_preds.argmax(dim=1).eq(target).float().mean().item()
        else:
            vis_preds = None
            vis_loss = torch.tensor(0.,requires_grad=True,device=device)

        vis_loss = vis_loss * self.alpha['vis']
        return vis_preds, vis_loss


    def fp_forward(self,h,target,log_vars,device,prefix=''):
        if self.losses_to_use['fp']:
            # input_ = self.maxpool(h.permute(0,2,1)).squeeze(-1)
            input_ = self.get_pooled_feats(h)
            fp_preds = self.fp_head(input_).squeeze(1)
            target = ( target > 9 ).float()
            fp_loss = self.bce(fp_preds, target)

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'fp_loss'] = fp_loss.item()
                log_vars[prefix+'fp_acc'] = (nn.Sigmoid()(fp_preds) > 0.5).float().eq(target).float().mean().item()
        else:
            fp_preds = None
            fp_loss = torch.tensor(0.,requires_grad=True,device=device)

        fp_loss = fp_loss * self.alpha['fp']
        return fp_preds, fp_loss

    def match_forward(self,h1,h2,match,log_vars,device,prefix=''):
        if self.losses_to_use['match']:
            if self.match_type == 'xcorr':

                match_in = self.xcorr(h1,h2)
                # match_in = self.maxpool(match_in.permute(0,2,1)
                match_in = self.get_pooled_feats(match_in)
                match_preds = self.match_head(match_in).squeeze(1)
                match_loss = self.bce(match_preds,match)

            elif self.match_type == 'xcorr_eff':

                match_in = self.xcorr_eff(h1,h2,self.combine)
                match_in = self.get_pooled_feats(match_in)
                match_preds = self.match_head(match_in).squeeze(1)
                match_loss = self.bce(match_preds,match)

            elif self.match_type == 'concat':
                cat = torch.cat([self.get_pooled_feats(h1),
                                 self.get_pooled_feats(h2)],dim=1)
                match_preds = self.match_head(cat).squeeze(1)
                match_loss = self.bce(match_preds,match)
            else:
                raise NotImplementedError

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'match_loss'] = match_loss.item()
                log_vars[prefix+'match_acc'] = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match).float().mean().item()

                gt_bins = torch.bincount(match.long())
                log_vars[prefix+'num_preds_0'] = gt_bins[0].item()
                log_vars[prefix+'num_preds_1'] = gt_bins[1].item() if len(gt_bins) > 1 else 0

                pred_bins = torch.bincount((nn.Sigmoid()(match_preds) > 0.5).long())
                log_vars[prefix+'num_gt_0'] = pred_bins[0].item()
                log_vars[prefix+'num_gt_1'] = pred_bins[1].item() if len(pred_bins) > 1 else 0
        else:
            match_preds = None
            match_loss = torch.tensor(0.,requires_grad=True,device=device)

        match_loss = match_loss * self.alpha['match']
        return match_preds, match_loss



    def match_forward_inference(self,h1,h2):
        if self.match_type == 'xcorr':
            match_in = self.xcorr(h1,h2)
            match_in = self.get_pooled_feats(match_in)
            match_preds = self.match_head(match_in).squeeze(1)

        elif self.match_type == 'xcorr_eff':
            match_in = self.xcorr_eff(h1,h2,self.combine)
            match_in = self.get_pooled_feats(match_in)
            match_preds = self.match_head(match_in).squeeze(1)

        elif self.match_type == 'concat':
            cat = torch.cat([self.maxpool(h1.permute(0,2,1)).squeeze(-1),
                                self.maxpool(h2.permute(0,2,1)).squeeze(-1)],dim=1)
            match_preds = self.match_head(cat).squeeze(1)
        else:
            raise NotImplementedError

        return match_preds




    def get_kl_loss(self,h1,h2,match,log_vars,device,prefix=''):

        if self.losses_to_use['kl']:
            kl_loss = self.kl(self.lsmx(h1.reshape(h1.size(0),-1)),
                              self.lsmx(h2.reshape(h2.size(0),-1))).mean(dim=1)
            
            where_no_match = torch.where(match == 0)
            kl_loss[where_no_match] = kl_loss[where_no_match] * -1
            kl_loss = kl_loss[match==0].mean() + kl_loss[match==1].mean()
            

            if self.compute_summary and log_vars != None:
                log_vars[prefix+'kl_loss'] = kl_loss.item()
        else:
            kl_loss = torch.tensor(0.,requires_grad=True,device=device)

        kl_loss = kl_loss * self.alpha['kl']
        return kl_loss


    def get_triplet_loss(self,h1,h2,id1,id2,match,log_vars,device,prefix=''):

        if self.losses_to_use['triplet']:
            match_idx = torch.where(match == 1)[0]
            matches = id1[match_idx]

            h_cat = torch.cat([h1,h2],dim=0)

            id_ = torch.cat([id1,id2],dim=0)

            a, p, n = [], [], []
            for i in range(matches.size(0)):


                m,idx = matches[i],match_idx[i]

                sample_pool = torch.where(id_ != m)[0]
                neg_idx_to_use = torch.multinomial(torch.ones(sample_pool.size(0)), self.triplet_sample_num, replacement=(len(sample_pool) < self.triplet_sample_num)).to(device)
                neg_idx_to_use = sample_pool[neg_idx_to_use]

                a.append(torch.full((self.triplet_sample_num,),idx,device=device))
                p.append(torch.full((self.triplet_sample_num,),idx,device=device))
                n.append(neg_idx_to_use)

            a = torch.cat(a,dim=0)
            p = torch.cat(p,dim=0)
            n = torch.cat(n,dim=0)

            a = h1.reshape(h1.size(0),-1)[a,...]
            p = h2.reshape(h1.size(0),-1)[p,...]
            n = h_cat.reshape(h_cat.size(0),-1)[n,...]


            triplet_loss = self.triplet_loss(anchor=a,
                                             positive=p,
                                             negative=n)


            if self.compute_summary and log_vars != None:
                log_vars[prefix+'triplet_loss'] = triplet_loss.item()
        else:
            triplet_loss = torch.tensor(0.,requires_grad=True,device=device)

        triplet_loss = triplet_loss * self.alpha['triplet']
        return triplet_loss


    def get_pooled_feats(self,h_cat):
        if self.pool_type == 'max':
            return self.maxpool(h_cat.permute(0,2,1)).squeeze(-1)
        elif self.pool_type == 'both':
            x1 = F.adaptive_max_pool1d(h_cat, 1).view(h_cat.size(0), -1)
            x2 = F.adaptive_avg_pool1d(h_cat, 1).view(h_cat.size(0), -1)
            return torch.cat((x1, x2), 1)
        else:
            raise NotImplementedError
    
    
    def preprocess_inputs(self, sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        vis_1 = torch.stack(vis_1,dim=0)
        vis_2 = torch.stack(vis_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)
        return sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2



    def preprocess_inputs_size(self, sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2,size_1,size_2,):
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        vis_1 = torch.cat(vis_1,dim=0)
        vis_2 = torch.cat(vis_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        id_1 = torch.cat(id_1,dim=0)
        id_2 = torch.cat(id_2,dim=0)
        size_1 = torch.cat(size_1,dim=0)
        size_2 = torch.cat(size_2,dim=0)
        return sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2,size_1,size_2


    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # print(data_batch)
        # exit(0)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    
    def forward_train(self,sparse_1,sparse_2,label_1,label_2,vis_1,vis_2,id_1,id_2):
        if self.eval_only:
            exit(0)

        log_vars = {}
        losses = {}

        sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2 = \
            self.preprocess_inputs(sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2)
        device = sparse_1.device
        
        #siamese_forward
        h1, h2 = self.siamese_forward(sparse_1,sparse_2)
        h_cat = torch.cat([h1,h2],dim=0)

        #cls forward
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')
        
        #vis forward
        vis_preds, vis_loss = self.vis_forward(h_cat,
                                               torch.cat([vis_1,vis_2],dim=0),
                                               torch.cat([id_1,id_2],dim=0),
                                               log_vars,device,prefix='')

        #FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        h1, h2, match = self.get_match_supervision(h1,h2,id_1,id_2)
        b,c,s = h_cat.shape
        temp = self.downsample(h_cat.reshape(-1,c)).reshape(b,self.downsample_dim,s)
        h1d, h2d = temp[:h1.size(0),...], temp[h1.size(0):,...]
        #match forward
        match_preds, match_loss = self.match_forward(h1d, h2d,match,log_vars,device,prefix='')
        #kl forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')
        #triplet forward
        triplet_loss = self.get_triplet_loss(h1d,h2d,id_1,id_2,match,log_vars,device,prefix='')

        losses['reid_loss'] = match_loss + vis_loss + cls_loss + kl_loss + fp_loss + triplet_loss

        return losses, log_vars


    def forward_test(self,sparse_1,sparse_2,label_1,label_2,vis_1,vis_2,id_1,id_2,size_1,size_2,*args,**kwargs):
        results = {}
        log_vars = None

        
        sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2,size_1,size_2 = \
            self.preprocess_inputs_size(sparse_1,sparse_2,vis_1,vis_2,label_1,label_2,id_1,id_2,size_1,size_2)
        device = sparse_1.device

        
        #siamese_forward
        h1, h2 = self.siamese_forward(sparse_1,sparse_2)
        h_cat = torch.cat([h1,h2],dim=0)

        #cls forward
        cls_preds, cls_loss = self.cls_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')
        
        #vis forward
        vis_preds, vis_loss = self.vis_forward(h_cat,
                                               torch.cat([vis_1,vis_2],dim=0),
                                               torch.cat([id_1,id_2],dim=0),
                                               log_vars,device,prefix='')

        #FP Forward
        fp_preds, fp_loss = self.fp_forward(h_cat,torch.cat([label_1,label_2],dim=0),log_vars,device,prefix='')

        h1, h2, match = self.get_match_supervision(h1,h2,id_1,id_2)
        b,c,s = h_cat.shape
        temp = self.downsample(h_cat.reshape(-1,c)).reshape(b,self.downsample_dim,s)
        h1d, h2d = temp[:h1.size(0),...], temp[h1.size(0):,...]
        #match forward
        match_preds, match_loss = self.match_forward(h1d, h2d,match,log_vars,device,prefix='')
        #kl forward
        kl_loss = self.get_kl_loss(h1,h2,match,log_vars,device,prefix='')
        #triplet forward
        triplet_loss = self.get_triplet_loss(h1d,h2d,id_1,id_2,match,log_vars,device,prefix='')


        results['val_fp_loss'] = torch.tensor([fp_loss])
        results['val_match_loss'] = torch.tensor([match_loss])
        results['val_cls_loss'] = torch.tensor([cls_loss])
        results['val_kl_loss'] = torch.tensor([kl_loss])
        results['val_triplet_loss'] = torch.tensor([triplet_loss])
        results['val_vis_loss'] = torch.tensor([vis_loss])
        results['val_match_preds'] = match_preds
        results['val_match_gt'] = match
        results['val_cls_preds'] = cls_preds
        results['val_cls_gt'] = torch.cat([label_1,label_2],dim=0)
        vis_filter = torch.where(torch.logical_and(torch.cat([id_1,id_2],dim=0) != -1,
                                                   torch.cat([vis_1,vis_2],dim=0) != -1))
        results['val_vis_preds'] = vis_preds
        results['val_vis_gt'] = torch.cat([vis_1,vis_2],dim=0)[vis_filter]
        # assert  results['val_vis_gt'].size(0) == results['val_vis_preds'].size(0), "shape not match {} vs {} ".format(results['val_vis_gt'].shape,results['val_vis_preds'].shape)
        results['val_fp_preds'] = fp_preds
        results['val_fp_gt'] = ( torch.cat([label_1,label_2],dim=0) > 9 ).float()
        results['match_classes'] = torch.cat([label_1.unsqueeze(1),label_2.unsqueeze(1)],dim=1)

        results['val_vis_gt_all'] = torch.cat([vis_1.unsqueeze(1),vis_2.unsqueeze(1)],dim=1)
        results['num_points'] = torch.cat([size_1.unsqueeze(1),size_2.unsqueeze(1)],dim=1)

        return [ results ] 




    #MMDET3d API

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        
        if self.verbose:
            t1 = time.time()
            print("{} Starting train_step()".format(self.log_msg()))


        losses, log_vars_train = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars.update(log_vars_train)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))
        
        if self.verbose:
            print("{} Ending train_step() after {}s".format(self.log_msg(),time.time()-t1))

        return outputs




    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))

        return outputs


    # def train_step(self, data_batch, optimizer,*args, **kwargs):
    #     return self.forward(data_batch,*args, **kwargs)


    def extract_feat(self,*args,**kwargs):
        raise NotImplementedError

    def show_result(self):
        raise NotImplementedError

    def aug_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def simple_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def init_weights(self):
        pass
