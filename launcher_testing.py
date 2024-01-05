import os
import time


seconds_per_hour = 60 * 60
hours_to_wait = 0#6
seconds_to_wait = hours_to_wait * seconds_per_hour

t1 = time.time()
print("waiting for {} seconds".format(seconds_to_wait))
time.sleep(seconds_to_wait)
print(time.time()-t1)

configs = [
    #IMAGE REID --- NUSCENES
    # ('testing_rgb_deit-base_pt_nus_det_200e.py','rgb_deit-base_pt_nus_det_200e.pth','nuscenes',),
    # ('testing_rgb_deit-tiny_pt_nus_det_200e.py','rgb_deit-tiny_pt_nus_det_200e.pth','nuscenes',),
    # ('testing_rgb_deit-tiny_r_nus_det_500e.py','rgb_deit-tiny_r_nus_det_500e.pth','nuscenes',),

    # Point REID --- NUSCENES
    # ('testing_pts_pointnet_r_nus_det_500e.py','pts_pointnet_r_nus_det_500e.pth','nuscenes',),
    # ('testing_pts_point-transformer_r_nus_det_500e.py','pts_point-transformer_r_nus_det_500e.pth','nuscenes',),
    # ('testing_pts_dgcnn_r_nus_det_500e.py','pts_dgcnn_r_nus_det_500e.pth','nuscenes',),

    # Point BASELINE --- NUSCENES
    # ('testing_pts_point-transformer_baseline-stnet_r_nus_det_500e.py','pts_point-transformer-baseline-stnet_r_nus_det_500e.pth','nuscenes',),

    #Scaling  --- NUSCENES
    # ('testing_pts_point-transformer_r_nus_det_500e.py','pts_point-transformer_r_nus_det_1000e.pth','scaling-nuscenes',),
    # ('testing_pts_point-transformer_r_nus_det_500e.py','pts_point-transformer_r_nus_det_2000e.pth','scaling-nuscenes',),
    ('testing_pts_point-transformer_r_nus_det_500e.py','pts_point-transformer_r_nus_det_4000e.pth','scaling-nuscenes',),

]

neptune_prefix = "no_point_filter_"

GPUS = [0]
neptune_prefix = None
prefix =  "CUDA_VISIBLE_DEVICES={} CUDA_LAUNCH_BLOCKING=1 ".format(
        ','.join([str(x) for x in GPUS])
    )
for c,ckpt,folder in configs:
    if 'rgb_' in c:
        if 'waymo' in c:
            dir_ = 'reid_waymo_image'
        elif 'nus' in c:
            dir_ = 'reid_nuscenes_image'
    else:
        if 'waymo' in c:
            dir_ = 'reid_waymo_pts'
        elif 'nus' in c:
            dir_ = 'reid_nuscenes_pts'

    
    command = prefix + "MASTER_ADDR=localhost torchpack dist-run -v -np {} python tools/train.py configs_reid/{}/{} --checkpoint pretrained/{}/{}".format(
        len(GPUS),dir_,c,folder,ckpt)

    if neptune_prefix is not None:
        command += "  --neptune-prefix {}".format(neptune_prefix)
    print(command)
    os.system(command)
