_base_ = [
    "./testing_base.py",
    "../_base_/reidentifiers/image/reid_image_deit-tiny_point-cat.py",
    "../_base_/schedules/cyclic_500e_lr1e-5.py",
]

neptune_tags = ['nus','only-match','testing','random init','deit-tiny']

model=dict(
    eval_only=True,
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
