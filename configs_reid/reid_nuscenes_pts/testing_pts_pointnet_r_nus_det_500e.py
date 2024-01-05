_base_ = [
    "./testing_base.py",
    "../_base_/reidentifiers/reid_pts_pointnet_point-cat.py",
    "../_base_/schedules/cyclic_500e_lr1e-5.py",
]

neptune_tags = ['nus','only-match','testing','Pointnet','point-cat']

model=dict(
    eval_only=True,
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
        vis=1,
        triplet=1,),
    triplet_loss=dict(margin=10,p=2),
    triplet_sample_num=32,
)


# dataloader_kwargs = dict(shuffle=True, prefetch_factor=2,persistent_workers=True)
# evaluation = dict(interval=1, pipeline=[], start=0)
# load_from='pretrained/reid/only-match/pts_pointnet_r_nus_det_500e.pth'