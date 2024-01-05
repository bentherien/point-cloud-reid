import os.path as osp


def setup_neptune_logger(cfg,args,neptune_prefix=None,checkpoint=None):
    try:
        cfg.dataloader_kwargs
    except AttributeError:
        raise AttributeError("You need to add 'dataloader_kwargs' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")

    try:
        cfg.train_tracker
    except AttributeError:
        raise AttributeError("You need to add 'train_tracker' to your config file. See 'configs/_base_/reidentification_runtine.py' for an example.")
        

    if cfg.train_tracker:
        assert cfg.dataloader_kwargs['shuffle'] == False, "You need to set 'dataloader_kwargs.shuffle' to False when training a tracker."

 
    try:
        if cfg.log_config.hooks[1].type == 'NeptuneLoggerHook':
            source_files = [x.strip() for x in cfg._text.split("\n") if x.strip().endswith(".py")]
            cfg.log_config.hooks[1].init_kwargs['source_files'] = source_files
            
            schedule_file = [x for x in source_files if "schedule" in x][0]
            dataset_file = [x for x in source_files if "/datasets/" in x][0]

            if 'medium-balaced' in dataset_file:
                dataset = 'v1.0-medium-balaced'
            elif 'medium' in dataset_file:
                dataset = 'v1.0-medium'
            elif 'mini' in dataset_file:
                dataset = 'v1.0-mini'
            else:
                dataset = 'v1.0-trainval'


            if cfg.log_config.hooks[1].init_kwargs.project == "bentherien/re-identification":
                neptune_name = "[Schedule] {} [Config] {}".format(osp.basename(schedule_file), 
                                                                                        osp.basename(args.config),)
            else:
                model_file = [x for x in source_files if "pnp_net" in x][-1]
                neptune_name = "[Schedule] {} [Model] {} [Config] {} [gpus] {}".format(osp.basename(schedule_file), 
                                                                                        osp.basename(model_file), 
                                                                                        osp.basename(args.config),
                                                                                        cfg.data.train.gpus)
                                                                                        
            # print(neptune_name)
            # exit(0)

            if checkpoint is not None:
                neptune_name = "[checkpoint] " + checkpoint + ' ' + neptune_name

            if neptune_prefix is not None:
                neptune_name = neptune_prefix + '_' + neptune_name
                
            cfg.log_config.hooks[1].init_kwargs['name'] += neptune_name
            cfg.log_config.hooks[1].init_kwargs['tags'] += [dataset] + cfg.neptune_tags + ['final testing - even', 'eval-flip' ]#['new-testing'+'waymo-pts']
            

            print(cfg.log_config.hooks[1].init_kwargs['name'])

            
        else:
            print('###############################################################################################')
            print('\t WARNING : No NeptuneLoggerHook in config file. This run will not be logged to Neptune.')
            print('###############################################################################################')
            time.sleep(3)

    except IndexError:

        pass

    return cfg