import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, generate_text, evaluate_result
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
import mmcv
from apex import amp
from utils.config import get_config
from models import xclip
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    # model parameters
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.001, type=float, help='weight of sparse loss')
    # attack parameters
    parser.add_argument('--eps', default=2/255, type=float, help='epsilon')

    args = parser.parse_args()

    config = get_config(args)

    return args, config

def main(config, eps, num_attack_steps=10):
    train_data, val_data, test_data, train_loader, val_loader, test_loader, val_loader_train, _ = build_dataloader(logger, config)
    model, _, model_path = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                            device="cpu", jit=False, 
                            T=config.DATA.NUM_FRAMES,
                            droppath=config.MODEL.DROP_PATH_RATE, 
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                            use_cache=config.MODEL.FIX_TEXT,
                            logger=logger,
                            )
    model = model.cuda()
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    
    step_size = 2.5 * (eps / num_attack_steps)
    chunk_size = 16
    
    vid_list = get_vid_list(config)
    scores_dict = dict()
    scores_dict['prd'] = dict()    

    text_labels = generate_text(train_data)
    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = batch_data["imgs"].cuda()
        label_id = batch_data["label"].cuda()
        bz = images.shape[0]
        n_clips = images.shape[1]    
        
        images = rearrange(images, 'b n c t h w -> (b n) t c h w')
            
        # logger.info(f"Size of batch data: {images.shape}, {label_id}")
        
        original_images = images.clone().detach()
        adv_images = images.clone().detach()        
        
        # save_image(original_images[0][0], f"original_{idx}")
        
        for step in range(num_attack_steps):
            adv_images.requires_grad_()
            
            output = model(adv_images, texts)
            logits = rearrange(output['y'], '(b k) c -> b k c', b=bz)
            scores = F.softmax(logits, dim=-1)
            scores_ano = scores[:,:,1]
            max_prob_ano, max_ind = torch.max(scores_ano, dim=-1)
            
            logits_video = torch.gather(logits, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1)
            max_prob_video, _ = torch.max(torch.gather(scores, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1),
                                        dim=-1)
            labels_binary = label_id > 0
            
            loss_mil = F.cross_entropy(logits_video, labels_binary.long(), reduction='none')
            loss_mil = loss_mil * max_prob_video
            loss_mil = loss_mil.mean()

            scores_all = scores
            smoothed_scores = (scores_all[:,1:,1] - scores_all[:,:-1,1])
            smoothed_loss = smoothed_scores.pow(2).sum(dim=-1).mean()

            sparsity_loss = scores_all[:,:,1].sum(dim=-1).mean()

            w_smooth = args.w_smooth
            w_sparse = args.w_sparse

            total_loss = loss_mil + smoothed_loss * w_smooth + sparsity_loss * w_sparse

            total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS
        
            total_loss.backward()

            grad_sign = adv_images.grad.sign()
            adv_images = adv_images.detach() + step_size * grad_sign
            perturbation = torch.clamp(adv_images - original_images, min=-eps, max=eps)
            adv_images = torch.clamp(original_images + perturbation, 0, 1)
        
        # save_image(adv_images[0][0], f"adv_{idx}")
        
        with torch.no_grad():
            final_outputs = model(adv_images, texts)
            
            final_scores = F.softmax(final_outputs['y'], dim=-1)
            final_scores = rearrange(final_scores, '(b k) c -> b k c', b=bz)
            final_scores_np_prd = final_scores.cpu().data.numpy()

            for ind in range(final_scores_np_prd.shape[0]):
                v_name = vid_list[batch_data["vid"][ind]]
                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                scores_dict['prd'][v_name].append(final_scores_np_prd[ind])
    
    tmp_dict = {}
    for v_name in scores_dict["prd"].keys():
        p_scores = np.array(scores_dict["prd"][v_name]).copy()
        if p_scores.shape[0] == 1:
            # 1,T,2
            tmp_dict[v_name] = [p_scores[0, :, 1]]
        else:
            # T,1,2
            tmp_dict[v_name] = [p_scores[:, 0, 1]]

    auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

    logger.info(
        f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t'
    )

def get_gt(config):
    GT = []
    videos = {}
    for video in open(config.DATA.VAL_FILE):
        vid = video.strip().split(' ')[0].split('/')[-1]
        video_len = int(video.strip().split(' ')[1])
        sub_video_gt = np.zeros((video_len,), dtype=np.int8)
        anomaly_tuple = video.split(' ')[3:]
        for ind in range(len(anomaly_tuple) // 2):
            start = int(anomaly_tuple[2 * ind])
            end = int(anomaly_tuple[2 * ind + 1])
            if start > 0:
                sub_video_gt[start:end] = 1
        videos[vid] = sub_video_gt
        
    return videos

def get_vid_list(config):
    vid_list = []
    with open(config.DATA.VAL_FILE, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)
    return vid_list

def save_image(image, name):
    image = image.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    path = os.path.join(config.OUTPUT, f"{name}.png")
    plt.imsave(path, image)
    logger.info(f"Saved image to {path}")


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"Working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config, args.eps)
