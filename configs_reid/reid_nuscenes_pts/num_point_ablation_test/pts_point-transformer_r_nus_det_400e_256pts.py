_base_ = [
    "../testing_pts_point-transformer_r_nus_det_500e.py",
]

neptune_tags = ['256 pts','nuscenes','only-match','Point Transformer','point-cat']

model=dict(
    backbone_list=[256, 128, 64]
)

data=dict(train=dict(subsample_sparse=256),
          val=dict(subsample_sparse=256,),)

