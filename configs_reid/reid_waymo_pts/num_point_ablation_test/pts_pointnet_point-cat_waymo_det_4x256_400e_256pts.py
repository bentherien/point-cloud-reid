_base_ = [
    "../testing_pts_pointnet_r_waymo_det_400e.py",
]

neptune_tags = ['256 pts','waymo','only-match','PointNet','point-cat']

model=dict(
    backbone_list=[256, 128, 64]
)

data=dict(train=dict(subsample_sparse=224),
          val=dict(subsample_sparse=224,),)




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
