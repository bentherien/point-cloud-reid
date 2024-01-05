_base_ = [
    "./testing_base.py",
    "../_base_/reidentifiers/image/reid_image_deit-tiny_point-cat.py",
    "../_base_/schedules/cyclic_200e_lr1e-5.py",
]

neptune_tags = ['waymo','only-match','testing','deit-tiny']

model=dict(
    eval_only=True,
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




# dataloader_kwargs = dict(shuffle=True, prefetch_factor=2,persistent_workers=True)
# evaluation = dict(interval=1, pipeline=[], start=0)
# load_from='pretrained/reid/only-match/rgb_deit-tiny_pt_nus_det_200e.pth'