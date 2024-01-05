tracking_classes = {
    'vehicle.bicycle':'bicycle',
    'vehicle.bus.bendy':'bus',
    'vehicle.bus.rigid':'bus',
    'vehicle.car':'car',
    'vehicle.motorcycle':'motorcycle',
    'human.pedestrian.adult':'pedestrian',
    'human.pedestrian.child':'pedestrian',
    'human.pedestrian.construction_worker':'pedestrian',
    'human.pedestrian.police_officer':'pedestrian',
    'vehicle.trailer':'trailer',
    'vehicle.truck':'truck',
    'bicycle':'bicycle',
    'truck':'truck',
    'car':'car',
    'trailer':'trailer',
    'bus':'bus',
    'motorcycle':'motorcycle',
    'pedestrian':'pedestrian'
}
                    
tracking_classes_fp = {
    'bicycle':'bicycle',
    'truck':'truck',
    'car':'car',
    'trailer':'trailer',
    'bus':'bus',
    'motorcycle':'motorcycle',
    'pedestrian':'pedestrian'
}

cls_to_idx = {
    'none_key':-1,
    'car':0, 'truck':1, 
    'construction_vehicle':2, 'bus':3, 
    'trailer':4, 'barrier':5, 
    'motorcycle':6, 'bicycle':7, 
    'pedestrian':8, 'traffic_cone':9
}

cls_to_idx_fp = {
    'none_key':-1,
    'car':0, 
    'truck':1, 
    'construction_vehicle':2, 
    'bus':3, 
    'trailer':4, 
    'barrier':5, 
    'motorcycle':6, 
    'bicycle':7, 
    'pedestrian':8, 
    'traffic_cone':9,
    'FP_car':10, 
    'FP_truck':11, 
    'FP_construction_vehicle':12, 
    'FP_bus':13, 
    'FP_trailer':14, 
    'FP_barrier':15, 
    'FP_motorcycle':16, 
    'FP_bicycle':17, 
    'FP_pedestrian':18, 
    'FP_traffic_cone':19
}


CLASSES = ['car','truck', 'construction_vehicle', 'bus', 'trailer','barrier', 'motorcycle','bicycle','pedestrian', 'traffic_cone',]

version = 'trainval'
train_metadata_version = 'trainval-det-both'
val_metadata_version = 'trainval-det-both'
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(type='ReIDDatasetNuscenesFP',
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
               sparse_loader=dict(type='ObjectLoaderSparseNuscenes',
                                train=True,
                                version='v1.0-{}'.format(version),
                                tracking_classes=tracking_classes,
                                metadata_path='data/lstk/sparse-{}/metadata/metadata.pkl'.format(train_metadata_version),
                                data_root='data/lstk/sparse-{}'.format(train_metadata_version),
                                min_points=2,
                                load_scene=True,
                                load_objects=True,
                                load_feats=['xyz'],
                                load_dims=[3],
                                to_ego_frame=False,
                                filter_mode='pts',
                                use_distance=True),

               complete_loader=dict(type='FakeCompleteLoader',)
            #    complete_loader=dict(type='ObjectLoaderCompleteNuscenes',
            #                         train=True,
            #                         version='v1.0-{}'.format(version),
            #                         metadata_path='data/lstk/complete/nuscenes/metadata.pkl',
            #                         data_root='data/lstk/complete/nuscenes',
            #                         load_scene=True,
            #                         load_objects=True,
            #                         load_feats=['xyz'],
            #                         load_dims=[3],
            #                         to_ego_frame=False,)
            ),
    val=dict(type='ReIDDatasetNuscenesFPValEven',
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
               sparse_loader=dict(type='ObjectLoaderSparseNuscenes',
                                train=False,
                                version='v1.0-{}'.format(version),
                                tracking_classes=tracking_classes,
                                metadata_path='data/lstk/sparse-{}/metadata/metadata.pkl'.format(train_metadata_version),
                                data_root='data/lstk/sparse-{}'.format(train_metadata_version),
                                min_points=2,
                                load_scene=True,
                                load_objects=True,
                                load_feats=['xyz'],
                                load_dims=[3],
                                to_ego_frame=False,
                                filter_mode='pts and vis',
                                use_distance=True),

               complete_loader=dict(type='FakeCompleteLoader',))
            #    complete_loader=dict(type='ObjectLoaderCompleteNuscenes',
            #                         train=False,
            #                         version='v1.0-{}'.format(version),
            #                         metadata_path='data/lstk/complete/nuscenes/metadata.pkl',
            #                         data_root='data/lstk/complete/nuscenes',
            #                         load_scene=True,
            #                         load_objects=True,
            #                         load_feats=['xyz'],
            #                         load_dims=[3],
            #                         to_ego_frame=False,)
            # ),
)