_base_ = [
    "../pts_dgcnn_point-cat_waymo_det_4x256_400e_accum2.py",
]

neptune_tags = ['1024 pts','waymo','only-match','DGCNN','point-cat']

model=dict(
    backbone_list=[1024, 512, 256]
)

data=dict(train=dict(subsample_sparse=1024),
          val=dict(subsample_sparse=1024,),)


# [96, 48, 24] -32 -1
# torch.Size([64, 32, 256])
# [128, 64, 32] 0 0
# torch.Size([64, 32, 256])
# [160, 80, 40] 32 1
# torch.Size([64, 32, 256])
# [192, 96, 48] 64 2
# torch.Size([64, 32, 256])
# [224, 112, 56] 96 3
# torch.Size([64, 32, 256])
# [256, 128, 64] 128 4