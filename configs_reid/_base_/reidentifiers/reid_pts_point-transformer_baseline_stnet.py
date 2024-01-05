_base_ = ['./reid_pts_point-transformer_point-cat.py']


model = dict(
    match_type='xcorr-baseline',
    combine='point-cat',
    match_head=[dict(type='LinearRes', n_in=128, n_out=128, norm='GN',ng=8),
                dict(type='Linear', in_features=128, out_features=1)],
)
