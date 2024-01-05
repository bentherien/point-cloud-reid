_base_ = [
    "../_base_/datasets/reid_nuscenes_pts.py",
    "../_base_/schedules/cyclic_500e_lr3e-4_norm1_accum2.py",
    "../_base_/reidentification_runtime.py",
]

data=dict(samples_per_gpu=128,
          val_samples_per_gpu=512,
          workers_per_gpu=4,
          train=dict(subsample_sparse=128),
          val=dict(subsample_sparse=128,
                   max_combinations=2 ),)

evaluation = dict(interval=50, pipeline=[], start=0)
dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=2,persistent_workers=True),
                         train=dict(shuffle=True, prefetch_factor=16,persistent_workers=True,drop_last=True))

