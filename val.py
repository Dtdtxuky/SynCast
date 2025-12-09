import argparse
import os
import torch
from utils.builder import ConfigBuilder
import utils.misc as utils
import yaml
from utils.logger import get_logger
from megatron_utils.tensor_parallel.data import get_data_loader_length
import shutil
import copy


#----------------------------------------------------------------------------

def subprocess_fn(args):
    utils.setup_seed(args.seed * args.world_size + args.rank)

    split = args.split
    test_name = args.test_name
    logger = get_logger(f'{split}', args.run_dir, utils.get_rank(), filename=f'{test_name}.log', resume=args.resume)

    # args.logger = logger
    args.cfg_params["logger"] = logger
    logger.info(f'run dir: {args.run_dir}')
    # build config
    logger.info('Building config ...')
    ### using distributedsampler for save ###
    args.cfg_params['sampler']['type'] = 'DistributedSampler'
    builder = ConfigBuilder(**args.cfg_params)

    logger.info('Building dataloaders ...')
    save_dataloader = builder.get_dataloader(split = f'{split}')
    logger.info(f'{split} dataloaders build complete')
    
    # build model
    logger.info('Building models ...')
    model = builder.get_model()
    if args.ckpt!=None:
        print(args.ckpt)
        model.load_checkpoint(args.ckpt, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)

    model_without_ddp = utils.DistributedParallel_Model(model, args.local_rank)
    # ### load checkpoint ###
    # ckpt_path = os.path.join(args.run_dir.split('/')[-3], args.run_dir.split('/')[-2], args.run_dir.split('/')[-1], 'checkpoint_best.pth')
    # model_without_ddp.load_checkpoint(args.ckpt, load_model=True, load_optimizer=False, load_scheduler=False, load_epoch=False, load_metric_best=False)

    if args.world_size > 1:
        for key in model_without_ddp.model:
            utils.check_ddp_consistency(model_without_ddp.model[key])

    for key in model_without_ddp.model:
        params = [p for p in model_without_ddp.model[key].parameters() if p.requires_grad]
        cnt_params = sum([p.numel() for p in params])
        logger.info("params {key}: {cnt_params}".format(key=key, cnt_params=cnt_params))

    logger.info('begin saving ...')

    model_without_ddp.tester(save_dataloader, builder.get_max_epoch(), builder.get_max_step(), checkpoint_savedir=args.relative_checkpoint_dir if model_without_ddp.use_ceph else args.run_dir, resume=args.resume)
    # # model_without_ddp.save_sample(save_dataloader, args.split, start_step=0)
    # # model_without_ddp.vis_denoise(save_dataloader, args.split, start_step=0)
    # path = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/test/ddpm_val_ddim_sample.txt'
    # # model_without_ddp.save_sample(dataloader=save_dataloader, split=split)
    # model_without_ddp.get_win_loss_frame_samples_guidance()

     
def main(args):
    if args.world_size > 1:
        utils.init_distributed_mode(args)
    else:
        args.rank = 0
        # args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        torch.cuda.set_device(args.gpu)

    cfg_path = args.cfg
    args.cfgdir = os.path.dirname(cfg_path)
    run_dir = args.cfgdir
    print(run_dir)

    args.cfg = os.path.join(cfg_path)
    with open(args.cfg, 'r') as cfg_file:
        cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

    cfg_params['dataset']['test'] = copy.deepcopy(cfg_params['dataset']['valid'])

    if "checkpoint_path" in cfg_params["model"]["params"]["extra_params"]:
        del cfg_params["model"]["params"]["extra_params"]["checkpoint_path"]
    
    args.cfg_params = cfg_params
    args.run_dir = run_dir
    if "relative_checkpoint_dir" in cfg_params:
        args.relative_checkpoint_dir = cfg_params['relative_checkpoint_dir']

    print('Launching processes...')
    subprocess_fn(args)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_model_parallel_size', type = int,     default = 1,                                            help = 'tensor_model_parallel_size')
    parser.add_argument('--resume',                     action = "store_true",                                                  help = 'resume')
    parser.add_argument('--resume_from_config',         action = "store_true",                                                  help = 'resume from config')
    parser.add_argument('--seed',                       type = int,     default = 0,                                            help = 'seed')
    parser.add_argument('--cuda',                       type = int,     default = 0,                                            help = 'cuda id')
    parser.add_argument('--world_size',                 type = int,     default = 1,                                            help = 'Number of progress')
    parser.add_argument('--per_cpus',                   type = int,     default = 1,                                            help = 'Number of perCPUs to use')
    parser.add_argument('--local_rank',                 type=int,       default=0,                                              help='local rank')
    # parser.add_argument('--world_size',     type = int,     default = -1,                                           help = 'number of distributed processes')
    parser.add_argument('--init_method',                type = str,     default='tcp://127.0.0.1:23456',                        help = 'multi process init method')
    parser.add_argument('--outdir',                     type = str,     default='/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/experiments',  help = 'Where to save the results')
    parser.add_argument('--cfg', '-c',                  type = str,     default = '/mnt/shared-storage-user/xukaiyi/CodeSpace/rankcast/configs/BaseModel/test.yaml',      help = 'path to the configuration file')
    parser.add_argument('--desc',                       type=str,       default='STR',                                          help = 'String to include in result dir name')
    parser.add_argument('--visual_vars',                nargs='+',       default=None,                                          help = 'visual vars')
    # debug mode for quick debug #
    parser.add_argument('--debug', action='store_true', help='debug or not')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='resume checkpoint')
    parser.add_argument('--resume_cfg_file', type=str, default=None, help='resume from cfg file')
    # save args #
    parser.add_argument('--split', type=str, default='test', help='split')
    parser.add_argument('--test_name', type=str, default=None, help='test name')
    parser.add_argument('--cfg_path',         type = str,     default = '/mnt/petrelfs/xukaiyi/CodeSpace/rankcast/configs/DiffCast/tau_sevir_ddpm.yaml',  help = 'Where to save the results')
    parser.add_argument('--model',         type = str,     default = 'TAU',  help = 'Where to save the results')
    parser.add_argument('--ckpt',         type = str,     default = None,  help = 'Where to save the results')
    args = parser.parse_args()

    main(args)