_base_ = ['./reid_pts_point-transformer_point-cat.py']

output_sequence_size=64
model = dict(
    match_type='xcorr',
    combine='point-cat',
    match_head=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                dict(type='Linear', in_features=128, out_features=1)],
    local_stage1=dict(type='local_self_attention',d_model=output_sequence_size,nhead=2,attention='linear',knum=48,pos_size=output_sequence_size),
    local_stage2=dict(type='local_self_attention',d_model=output_sequence_size,nhead=2,attention='linear',knum=48,pos_size=output_sequence_size),
)
