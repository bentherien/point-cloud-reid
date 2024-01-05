_base_ = [
    # "../_base_/datasets/reidentificaiton-nus_det-train-kf_det-test_FP_errors.py",
    "../_base_/datasets/reid_nuscenes_pts.py",
    # "../_base_/datasets/reidentificaiton-nus_det-train-kf_det-test_FP_new.py",
    "../_base_/reidentification_runtime_testing.py",
]

data=dict(samples_per_gpu=128,
          val_samples_per_gpu=512,
          workers_per_gpu=1,
          val_workers_per_gpu=6,
          train=dict(subsample_sparse=128),
          val=dict(subsample_sparse=128,
                   max_combinations=10,
                #    sample_all='pts and vis',
                #    min_points=2,
                   sparse_loader=dict(min_points=2,filter_mode='pts and vis',)
                   
                #    err_filepath = '/home/datasets/lstk/reid_pairs/point_reid_reid_pairs.json',
                #    err_filepath = '/home/datasets/lstk/reid_pairs/probabilistic_tracking_reid_pairs.json',
                #    err_filepath = '/home/datasets/lstk/reid_pairs/simple_track_reid_pairs.json',
                   ),)


dataloader_kwargs = dict(val=dict(shuffle=True, prefetch_factor=36,persistent_workers=True),
                         train=dict(shuffle=True, prefetch_factor=18,persistent_workers=True))
evaluation = dict(interval=1, pipeline=[], start=0)


