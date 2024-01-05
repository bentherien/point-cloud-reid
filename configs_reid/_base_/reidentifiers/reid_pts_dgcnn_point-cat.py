num_classes  = 10 * 2
num_points = 2048
hidden_size = 128
ng = 16

output_sequence_size = 64

mul = 4 
hidden_size = output_sequence_size * 2
hidden_size_match = output_sequence_size * 2

downsample_dim = 64

downsample_input = 1024
model = dict(
    type='ReIDNet',
    hidden_size=hidden_size,
    pool_type='both',
    combine='point-cat',
    match_type='xcorr_eff',
    output_sequence_size=output_sequence_size,
    use_dgcnn=True,
    backbone_list=[128,64,32],

    backbone=dict(type='dgcnn',dropout=0.5,emb_dims=downsample_input, k=20, output_channels=40),

    # cls_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=ng),
    #           dict(type='Linear', in_features=hidden_size, out_features=num_classes)],
    
    # fp_head=[dict(type='LinearRes', n_in=hidden_size, n_out=hidden_size, norm='GN',ng=ng),
    #           dict(type='Linear', in_features=hidden_size, out_features=1)],

    match_head=[dict(type='LinearRes', n_in=hidden_size_match, n_out=hidden_size_match, norm='GN',ng=ng),
                dict(type='Linear', in_features=hidden_size_match, out_features=1)],
    
    # shape_head=[
    #     dict(type='Conv1d', in_channels=hidden_size, out_channels=1024, kernel_size=output_sequence_size//2),
    #     dict(type='BatchNorm1d', num_features=1024),
    #     dict(type='ReLU'),
    #     dict(type='Conv1d', in_channels=1024, out_channels=2048, kernel_size=output_sequence_size//4),
    #     dict(type='BatchNorm1d', num_features=2048),
    #     dict(type='ReLU'),
    #     dict(type='Conv1d', in_channels=2048, out_channels=2048, kernel_size=output_sequence_size//4),
    # ],
    
    cls_head=None,
    fp_head=None,
    shape_head=None,


    downsample=[dict(type='LinearRes', n_in=downsample_input, n_out=512, norm='GN',ng=64),
                dict(type='LinearRes', n_in=512, n_out=128, norm='GN',ng=16),
                dict(type='Linear', in_features=128, out_features=downsample_dim)],
    


    cross_stage1=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    cross_stage2=dict(type='corss_attention',d_model=output_sequence_size,nhead=2,attention='linear'),
    local_stage1=dict(),
    local_stage2=dict(),
)


                 