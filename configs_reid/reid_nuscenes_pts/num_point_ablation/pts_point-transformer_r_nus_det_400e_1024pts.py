_base_ = [
    "../pts_point-transformer_point-cat_nus_det_4x256_500e_accum2.py",
]
neptune_tags = ['1024 pts','nuscenes','only-match','Point Transformer','point-cat']

model=dict(
    backbone_list=[1024, 512, 256]
)

data=dict(train=dict(subsample_sparse=1024),
          val=dict(subsample_sparse=1024,),)

