_base_ = [
    "../pts_dgcnn_point-cat_waymo_det_4x256_400e.py",
]

neptune_tags = ['160 pts','waymo','only-match','DGCNN','point-cat']

model=dict(
    backbone_list=[160, 80, 40]
)

data=dict(train=dict(subsample_sparse=160),
          val=dict(subsample_sparse=160,),)





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