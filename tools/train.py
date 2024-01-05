import argparse
import copy
import os
import random
import time

import os.path as osp
import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

from tools.utils import setup_neptune_logger


def main():

    assert '/btherien/github/nuscenes-devkit/python-sdk' in os.environ['PYTHONPATH']
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help='the checkpoint file for evaluation',
    )
    parser.add_argument(
        "--neptune-prefix",
        type=str,
        default=None,
        help='the checkpoint file for evaluation',
    )
    args, opts = parser.parse_known_args()





    if 'configs_reid' in args.config.split('/')[0]:
        cfg = Config.fromfile(args.config)
        cfg = setup_neptune_logger(cfg,args,args.neptune_prefix,args.checkpoint)
        dataloader_kwargs=cfg.dataloader_kwargs
    else:
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg = Config(recursive_eval(configs), filename=args.config)
        dataloader_kwargs=dict(shuffle=True, prefetch_factor=4)
    

    if args.checkpoint is not None:
        print("loading from checkpoint:", args.checkpoint)
        cfg.load_from = args.checkpoint

    # print(cfg.pretty_text)
    # exit(0)


    # datasets = [build_dataset(cfg.data.val)]
    # exit(0)
    
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    print("\n\n")
    print("###############################################################################################")
    print("Setting local rank to {}".format(dist.local_rank()))
    print("###############################################################################################")
    print("\n\n")
    time.sleep(0.2)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # print(cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=cfg.validate,
        timestamp=timestamp,
        dataloader_kwargs=dataloader_kwargs,
    )


if __name__ == "__main__":
    main()
