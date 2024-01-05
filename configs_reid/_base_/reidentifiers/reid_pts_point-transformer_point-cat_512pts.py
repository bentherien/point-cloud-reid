_base_ = ['reid_pts_point-transformer_point-cat.py']

model = dict(
    backbone_list=[512,256,128],
)