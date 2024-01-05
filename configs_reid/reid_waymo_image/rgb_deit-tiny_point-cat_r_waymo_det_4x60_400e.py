_base_ = [
    "../_base_/reidentifiers/image/reid_image_deit-tiny_point-cat.py",
    "../_base_/schedules/cyclic_400e_lr1e-5_norm1_accum2.py",
    "./training_base.py",
]


model=dict(
    backbone='deit-tiny-no-pt',
    freeze_backbone=False,
    losses_to_use=dict(
        triplet=False,
        kl=False,
        match=True,
        cls=False,
        vis=False,
        fp=False,
    ),
    alpha=dict(
        kl=1,
        match=1,
        cls=1,
        shape=1,
        fp=1,
        vis=1,
        triplet=1,),
    triplet_loss=dict(margin=10,p=2),
    triplet_sample_num=32,
)

neptune_tags = ['500e','4 x 60','deit-tiny','images','random init','only match']
