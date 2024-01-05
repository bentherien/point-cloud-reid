num_classes  = 10 * 2
num_points = 2048
hidden_size = 128
ng = 32

output_sequence_size = 64

mul = 4 
hidden_size = output_sequence_size * 2
hidden_size_match = output_sequence_size * 4
# 2 for avg and maxpool and 2 for concatenating the features of each object


model = dict(
    type='ReIDNet',
    hidden_size=hidden_size,
    combine='cat',
    match_type='concat',
    output_sequence_size=output_sequence_size,
    backbone=dict(type='Pointnet_Backbone',input_channels=0,use_xyz=True,conv_out=output_sequence_size),
    backbone_list=[128,64,32], 
    pool_type='max',
    downsample=None,
    cls_head=None,
    fp_head=None,

    match_head=[dict(type='LinearRes', n_in=hidden_size_match, n_out=hidden_size_match, norm='GN',ng=ng),
                dict(type='Linear', in_features=hidden_size_match, out_features=1)],
    
    shape_head=[
        dict(type='Conv1d', in_channels=hidden_size, out_channels=1024, kernel_size=output_sequence_size//2),
        dict(type='BatchNorm1d', num_features=1024),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=1024, out_channels=2048, kernel_size=output_sequence_size//4),
        dict(type='BatchNorm1d', num_features=2048),
        dict(type='ReLU'),
        dict(type='Conv1d', in_channels=2048, out_channels=2048, kernel_size=output_sequence_size//4),
    ],
    cross_stage1=None,
    cross_stage2=None,
    local_stage1=None,
    local_stage2=None,
)


                 