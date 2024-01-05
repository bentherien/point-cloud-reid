import os
import time


seconds_per_hour = 60 * 60
hours_to_wait = 0.
seconds_to_wait = hours_to_wait * seconds_per_hour

t1 = time.time()
print("waiting for {} seconds".format(seconds_to_wait))
time.sleep(seconds_to_wait)
print(time.time()-t1)



configs = [

    #IMAGE REID --- NUSCENES
    ('reid_nuscenes_image','rgb_deit-base_point-cat_pt_nus_det_4x60_200e.py',66),
    # ('reid_nuscenes_image','rgb_deit-tiny_point-cat_pt_nus_det_4x60_200e.py',66),
    # ('reid_nuscenes_image','rgb_deit-tiny_point-cat_r_nus_det_4x60_500e.py',66),

    # Point REID --- NUSCENES
    # ('reid_nuscenes_pts','pts_point-transformer_point-cat_nus_det_4x256_500e.py',66),
    # ('reid_nuscenes_pts','pts_dgcnn_point-cat_nus_det_4x256_500e.py',66),
    # ('reid_nuscenes_pts','pts_pointnet_point-cat_nus_det_4x256_500e.py',66),

    # Point BASELINE --- NUSCENES
    # ('reid_nuscenes_pts','pts_point-transformer_baseline-stnet_nus_det_4x256_500e.py',66),
]

GPUS = [1]
neptune_prefix = None
prefix =  "CUDA_VISIBLE_DEVICES={} CUDA_LAUNCH_BLOCKING=1 ".format(
        ','.join([str(x) for x in GPUS])
    )

for dir_,c,seed in configs:
    if 'rgb_' in c:
        if 'waymo' in c:
            dir_ = 'reid_waymo_image'
        elif 'nus' in c:
            dir_ = 'reid_nuscenes_image'
    else:
        if 'waymo' in c:
            dir_ = 'reid_waymo_pts/num_point_ablation'
        elif 'nus' in c:
            dir_ = 'reid_nuscenes_pts'

    command = prefix + "MASTER_ADDR=localhost torchpack dist-run -v -np {} python tools/train.py configs_reid/{}/{} --seed {}".format(len(GPUS),dir_,c,seed)

    neptune_prefix = "seed:{}".format(seed)
    if neptune_prefix is not None:
        command += "  --neptune-prefix {}".format(neptune_prefix)

    print(command)
    os.system(command)
