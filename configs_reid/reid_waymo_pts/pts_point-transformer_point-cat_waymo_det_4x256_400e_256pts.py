_base_ = [
    "../_base_/reidentifiers/reid_pts_point-transformer_point-cat_256pts.py",
    "./training_base_400e_lr3e-4_4x256.py"
]

model=dict(
    losses_to_use=dict(
        kl=False,
        match=True,
        cls=False,
        shape=False,
        fp=False,
        triplet=False,
    ),
    alpha=dict(
        kl=1,
        match=1,
        cls=1,
        shape=1,
        fp=1,
        vis=0,
        triplet=1,),
    triplet_loss=dict(margin=10,p=2),
    triplet_sample_num=128,
)

neptune_tags = ['256 points input','waymo','400e','4 x 256','point-transformer','only match','point-cat']


data=dict(train=dict(subsample_sparse=256),
          val=dict(subsample_sparse=256),)