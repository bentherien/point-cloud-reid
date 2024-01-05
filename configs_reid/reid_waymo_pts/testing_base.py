_base_ = [
   #  "../_base_/datasets/reidentificaiton-waymo_det-test-even.py",
    "../_base_/datasets/reid_waymo_pts.py",
    "../_base_/reidentification_runtime_testing.py",
    # "../_base_/datasets/reidentificaiton-nus_det-train-kf_det-test_FP.py",
]

data=dict(samples_per_gpu=128,
          val_samples_per_gpu=512,
          workers_per_gpu=1,
          val_workers_per_gpu=8,
          train=dict(subsample_sparse=128),
          val=dict(subsample_sparse=128,
                   max_combinations=10,
                  #  min_points=2,
                  #  sample_all='pts and vis',#'pts'
                   sparse_loader=dict(min_points=2,filter_mode='pts and vis',)
                #    sample_all=True
                   ),
        )

dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=36,persistent_workers=True),
                         train=dict(shuffle=True, prefetch_factor=34,persistent_workers=True))
evaluation = dict(interval=1, pipeline=[], start=0)
# seed=66