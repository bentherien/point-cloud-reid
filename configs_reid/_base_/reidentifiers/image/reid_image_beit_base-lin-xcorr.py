num_classes  = 10 * 2
num_vis_classes = 4


hidden_size = 768
hidden_pred_size = hidden_size * 2
hiden_pred_ng = 64

downsample_dim = 32
hidden_match_size = 4 * downsample_dim
hidden_match_ng = 16
ng = 32

output_sequence_size = 32

model = dict(
    type='ImageReIDNet',
    dim=hidden_size,
    backbone='beit',
    downsample_dim=downsample_dim,
    downsample=[dict(type='LinearRes', n_in=hidden_size, n_out=256, norm='GN',ng=32),
                dict(type='LinearRes', n_in=256, n_out=128, norm='GN',ng=16),
                dict(type='Linear', in_features=128, out_features=downsample_dim)],

    cross_lin_attn=dict(type='cross_lin_attn',d_model=downsample_dim,nhead=2,attention='linear'),
    
    cls_head=[dict(type='LinearRes', n_in=hidden_pred_size, n_out=hidden_pred_size, norm='GN',ng=hiden_pred_ng),
              dict(type='Linear', in_features=hidden_pred_size, out_features=num_classes)],
    
    fp_head=[dict(type='LinearRes', n_in=hidden_pred_size, n_out=hidden_pred_size, norm='GN',ng=hiden_pred_ng),
              dict(type='Linear', in_features=hidden_pred_size, out_features=1)],

    vis_head=[dict(type='LinearRes', n_in=hidden_pred_size, n_out=hidden_pred_size, norm='GN',ng=hiden_pred_ng),
              dict(type='Linear', in_features=hidden_pred_size, out_features=num_vis_classes)],


    match_head=[dict(type='LinearRes', n_in=hidden_match_size, n_out=hidden_match_size, norm='GN',ng=hidden_match_ng),
                dict(type='Linear', in_features=hidden_match_size, out_features=1)],


)


                 