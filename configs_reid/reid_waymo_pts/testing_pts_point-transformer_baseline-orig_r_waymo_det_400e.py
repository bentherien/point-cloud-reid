_base_ = [
    "./testing_base.py",
    "../_base_/reidentifiers/reid_pts_point-transformer_baseline_orig.py",
    "../_base_/schedules/cyclic_400e_lr1e-5.py",
]

neptune_tags = ['waymo','only-match','testing','Baseline-orig']

model=dict(
    eval_flip=True,
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



# dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=36,persistent_workers=True),
#                          train=dict(shuffle=True, prefetch_factor=18,persistent_workers=True))
# evaluation = dict(interval=1, pipeline=[], start=0)
# load_from='pretrained/reid/only-match/pts_point-transformer_r_waymo_det_400e.pth'