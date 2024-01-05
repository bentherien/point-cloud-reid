_base_ = [
    "../pts_pointnet_point-cat_waymo_det_4x256_400e_accum2.py",
]

neptune_tags = ['2048 pts','waymo','only-match','PointNet','point-cat']


model=dict(
    backbone_list=[2048, 1024, 512]
)

data=dict(train=dict(subsample_sparse=2048),
          val=dict(subsample_sparse=2048,),)
