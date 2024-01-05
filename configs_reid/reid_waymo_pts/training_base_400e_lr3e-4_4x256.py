_base_ = [
    "../_base_/datasets/reid_waymo_pts.py",
    "../_base_/schedules/cyclic_400e_lr3e-4_norm1.py",
    "../_base_/reidentification_runtime.py",
]

data=dict(samples_per_gpu=256,
          val_samples_per_gpu=512,
          workers_per_gpu=4,
          train=dict(subsample_sparse=128),
          val=dict(subsample_sparse=128,
                   max_combinations=2 ),)

evaluation = dict(interval=50, pipeline=[], start=0)
dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=2,persistent_workers=True),
                         train=dict(shuffle=True, prefetch_factor=9,persistent_workers=True, drop_last=True))

