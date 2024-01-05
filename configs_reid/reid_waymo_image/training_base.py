_base_ = [
    "../_base_/datasets/reid_waymo_image.py",
    "../_base_/reidentification_runtime.py",
]

data=dict(samples_per_gpu=60,
          val_samples_per_gpu=512,
          workers_per_gpu=8,
          train=dict(subsample_sparse=128),
          val=dict(subsample_sparse=128,
                   max_combinations=2 ),)


evaluation = dict(interval=50, pipeline=[], start=0)
dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=2,persistent_workers=True),
                         train=dict(shuffle=True, prefetch_factor=27,persistent_workers=True,drop_last=True))
