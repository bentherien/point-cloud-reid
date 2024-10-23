mkdir pretrained &&
cd pretrained &&
mkdir nuscenes &&
cd nuscenes &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/rgb_deit-tiny_pt_nus_det_200e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_pointnet_r_nus_det_500e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_point-transformer_r_nus_det_500e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/rgb_deit-tiny_r_nus_det_500e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/rgb_deit-base_pt_nus_det_200e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_dgcnn_r_nus_det_500e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/pretrained/nuscenes/pts_point-transformer-baseline-stnet_r_nus_det_500e.pth &&
cd .. &&
mkdir scaling-nuscenes &&
cd scaling-nuscenes &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/scaling-nuscenes/pts_point-transformer_r_nus_det_4000e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/scaling-nuscenes/pts_point-transformer_r_nus_det_1000e.pth &&
wget https://wiselab.uwaterloo.ca/nuscenes-reid/scaling-nuscenes/pts_point-transformer_r_nus_det_2000e.pth
