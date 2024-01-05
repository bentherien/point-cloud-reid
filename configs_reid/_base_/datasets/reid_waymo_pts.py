CLASSES = ['car',
           'truck',
           'bus',
           'motorcycle',
           'bicycle',
           'pedestrian']

tracking_classes = {
    'bicycle':'bicycle',
    'truck':'truck',
    'car':'car',
    'trailer':'trailer',
    'bus':'bus',
    'motorcycle':'motorcycle',
    'pedestrian':'pedestrian'
}

tracking_classes_fp = tracking_classes

cls_to_idx = {
    'none_key':-1,
    'car':0,
    'truck':1,
    'bus':2,
    'motorcycle':3,
    'bicycle':4,
    'pedestrian':5
}

cls_to_idx_fp = {
    'none_key':-1,
    'car':0,
    'truck':1,
    'bus':2,
    'motorcycle':3,
    'bicycle':4,
    'pedestrian':5,
    'FP_car':6,
    'FP_truck':7,
    'FP_bus':8,
    'FP_motorcycle':9,
    'FP_bicycle':10,
    'FP_pedestrian':11,
}


dataset_root = 'data'
train_metadata_version = 'waymo-det-pts-train'
val_metadata_version = 'waymo-det-both-val'
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(type='ReIDDatasetWaymoFP',
               train=True,
               cls_to_idx=cls_to_idx,
               cls_to_idx_fp=cls_to_idx_fp,
               tracking_classes=tracking_classes,
               tracking_classes_fp=tracking_classes_fp,
               subsample_sparse=128,
               subsample_dense=2048,
               CLASSES=CLASSES,
               return_mode='dict',
               verbose=False,
               validation_seed=0,
               sparse_loader=dict(type='ObjectLoaderSparseWaymo',
                                metadata_path='{}/lstk/sparse-{}/metadata'.format(dataset_root,train_metadata_version),
                                data_root='{}/lstk/sparse-{}'.format(dataset_root,train_metadata_version),
                                min_points=2,
                                tracking_classes=tracking_classes,
                                load_scene=True,
                                load_objects=True,
                                load_feats=['xyz'],
                                load_dims=[3],
                                to_ego_frame=False,
                                filter_mode='pts',
                                use_distance=True),
               complete_loader=dict(type='FakeCompleteLoader',)

            #    complete_loader=dict(type='ObjectLoaderCompleteWaymo',
            #                         metadata_path='{}/lstk/complete/waymo/metadata/metadata_train_000-800.pkl'.format(dataset_root),
            #                         data_root='.',
            #                         load_scene=True,
            #                         load_objects=True,
            #                         load_feats=['xyz'],
            #                         load_dims=[3],
                                    # to_ego_frame=False,)
            ),
    val=dict(type='ReIDDatasetWaymoFPValEven',
               train=False,
               cls_to_idx=cls_to_idx,
               cls_to_idx_fp=cls_to_idx_fp,
               tracking_classes=tracking_classes,
               tracking_classes_fp=tracking_classes_fp,
               subsample_sparse=128,
               subsample_dense=2048,
               CLASSES=CLASSES,
               return_mode='dict',
               verbose=False,
               validation_seed=0,
               max_combinations=10,
               sparse_loader=dict(type='ObjectLoaderSparseWaymo',
                                metadata_path='{}/lstk/sparse-{}/metadata'.format(dataset_root,val_metadata_version),
                                data_root='{}/lstk/sparse-{}'.format(dataset_root,val_metadata_version),
                                min_points=2,
                                tracking_classes=tracking_classes,
                                use_metdata_fix=True,
                                load_scene=True,
                                load_objects=True,
                                load_feats=['xyz'],
                                load_dims=[3],
                                to_ego_frame=False,
                                filter_mode='pts and vis',
                                use_distance=True),

               complete_loader=dict(type='FakeCompleteLoader',)
            #    complete_loader=dict(type='ObjectLoaderCompleteWaymo',
            #                         metadata_path='{}/lstk/complete/waymo/metadata'.format(dataset_root),
            #                         data_root='.',
            #                         load_scene=True,
            #                         load_objects=True,
            #                         load_feats=['xyz'],
            #                         load_dims=[3],
            #                         to_ego_frame=False,)
            ),
)