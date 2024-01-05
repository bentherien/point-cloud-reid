num_classes  = 10 * 2
num_points = 2048
hidden_size = 128
ng = 32

output_sequence_size = 64

mul = 4 
hidden_size = output_sequence_size * 2
hidden_size_match = output_sequence_size * 2
# 2 for avg and maxpool and 2 for concatenating the features of each object


model = dict(
    type='ReIDNet',
    hidden_size=hidden_size,
    combine='point-cat',
    match_type='xcorr_eff',
    pool_type='both',
    backbone_list=[128,64,32],
    output_sequence_size=output_sequence_size,

    backbone=dict(type='Pointnet_Backbone',input_channels=0,use_xyz=True,conv_out=output_sequence_size),
    match_head=[dict(type='LinearRes', n_in=hidden_size_match, n_out=hidden_size_match, norm='GN',ng=8),
                dict(type='Linear', in_features=hidden_size_match, out_features=1)],

    # cls_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=16),
    #           dict(type='Linear', in_features=hidden_size, out_features=num_classes)],
    
    # fp_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=16),
    #           dict(type='Linear', in_features=hidden_size, out_features=1)],
    
    # shape_head=[
    #     dict(type='Conv1d', in_channels=hidden_size, out_channels=1024, kernel_size=output_sequence_size//2),
    #     dict(type='BatchNorm1d', num_features=1024),
    #     dict(type='ReLU'),
    #     dict(type='Conv1d', in_channels=1024, out_channels=2048, kernel_size=output_sequence_size//4),
    #     dict(type='BatchNorm1d', num_features=2048),
    #     dict(type='ReLU'),
    #     dict(type='Conv1d', in_channels=2048, out_channels=2048, kernel_size=output_sequence_size//4),
    # ],
    downsample=None,
    cls_head=None,
    fp_head=None,
    shape_head=None,
    
    cross_stage1=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    cross_stage2=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    local_stage1=dict(),
    local_stage2=dict(),
)


                 