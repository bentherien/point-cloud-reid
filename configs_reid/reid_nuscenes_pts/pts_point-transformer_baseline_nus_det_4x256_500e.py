_base_ = [
    "../_base_/reidentifiers/reid_pts_point-transformer_baseline.py",
    "./training_base_500e_lr3e-4_4x256.py"
]


model=dict(
    losses_to_use=dict(
        kl=False,
        match=True,
        cls=False,
        shape=False,
        fp=False,
        triplet=False,
        dense=False,
    ),
    alpha=dict(
        kl=1,
        match=1,
        dense=0,
        cls=1,
        shape=1,
        fp=1,
        vis=0,
        triplet=1,),
    triplet_loss=dict(margin=10,p=2),
    triplet_sample_num=128,
)

neptune_tags = ['nus','400e','4 x 256','point-transformer','baseline','only match']