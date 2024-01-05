# _base_ = ['./reid_small-lin-xcorr_fp.py']

num_classes  = 10 * 2
num_points = 2048
hidden_size = 128
ng = 32

output_sequence_size = 64

mul = 2
hidden_size = output_sequence_size * 2
hidden_size_match = output_sequence_size * 4
# 2 for avg and maxpool and 2 for concatenating the features of each object
model = dict(
    combine='point-cat',
    type='ReIDNet',
    hidden_size=hidden_size,
    match_type='xcorr_eff',
    output_sequence_size=output_sequence_size,
    backbone=dict(type='Pointnet_Backbone',input_channels=0,use_xyz=True,conv_out=output_sequence_size,mul=2),
    backbone_list=[128,64,32], 
    pool_type='both',
    downsample=None,
    cls_head=None,
    fp_head=None,
    shape_head=None,
    match_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=8),
                dict(type='Linear', in_features=hidden_size, out_features=1)],
    cross_stage1=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    cross_stage2=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    local_stage1=dict(),
    local_stage2=dict(),
)


                 