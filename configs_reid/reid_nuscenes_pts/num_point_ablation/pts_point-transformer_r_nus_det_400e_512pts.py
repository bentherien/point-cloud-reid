_base_ = [
    "../pts_point-transformer_point-cat_nus_det_4x256_500e.py",
]
neptune_tags = ['512 pts','nuscenes','only-match','Point Transformer','point-cat']

model=dict(
    backbone_list=[512, 256, 128]
)

data=dict(train=dict(subsample_sparse=512),
          val=dict(subsample_sparse=512,),)

