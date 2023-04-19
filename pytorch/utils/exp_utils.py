import functools
import os, shutil
from datetime import datetime
import random

import numpy as np
import wandb
import torch


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))

def init_wandb(run_name, run_cfg):
    # start a new wandb run to track this script

    wandb.init(
        project="fsdp",
        config=run_cfg,
        entity="ridzy619",
        name=run_name
    )
    
    if run_cfg.debug:
        wandb.log = get_logger(log_path=os.path.join(run_cfg.dir_path, 'wandb_local_metrics.txt'))
    return wandb
