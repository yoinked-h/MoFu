from model_handler import TokenizerModels
from dataloader import Dataset, SFDataset
from config import MoFUConfig #noqa
from pathlib import Path
import torch
from safetensors.torch import save_file

def dtp(x):
    match x:
        case 'fp16':
            return torch.float16
        case 'fp32':
            return torch.float32
        case 'bf16':
            return torch.bfloat16
        case _:
            return torch.float32
        
        

def get_useful_tags(dset: Dataset, cfg):
    usage = dset.get_tags_and_usage()
    #filter out the less common tags
    percent = cfg.dropout_percentile
    datasize = len(dset)
    useful = []
    for tag in usage.keys():
        if usage[tag] > round(datasize * percent):
            useful.append(tag)
    return useful

def create_MoFU(tags, cfg: MoFUConfig):
    model = TokenizerModels(cfg.diffusers_name, cfg.device)
    MoFU = model.encode(", ".join(tags))
    #save with safetensors
    tosave = {"model": MoFU}
    pathto = Path(cfg.output_dir + cfg.MoFU_name + ".safetensors")
    save_file(tosave, pathto, {"name": cfg.MoFU_name, "base_model": cfg.diffusers_name, "version": '1', "notes": cfg.notes})


def create_MoFUv2(tags, cfg: MoFUConfig):
    model = TokenizerModels(cfg.diffusers_name, cfg.device, cfg.sdxl)
    MoFus = []
    for tag in tags:
        MoFus.append(model.encode(tag))
    match cfg.merge_method:
        case "average":
            #add the average
            if not cfg.sdxl:
                mofusum = torch.zeros(MoFus[0].shape)
                for mofu in MoFus:
                    mofusum += mofu
                MoFu = mofusum / len(MoFus)
            else:
                mofusum = [torch.zeros(MoFus[0][0].shape, device=cfg.device), torch.zeros(MoFus[0][1].shape, device=cfg.device)]
                for mofu in MoFus:
                    for i in range(2):
                        mofusum[i] += mofu[i]
                MoFu = [None, None]
                for i in range(2):
                    MoFu[i] = mofusum[i] / len(MoFus)
        case _:
            print("unknown merge method")
            if not cfg.sdxl:
                MoFu = MoFus[0] 
            else:
                MoFu = [MoFus[0][0], MoFus[0][1]]
    #save with safetensors
    if not cfg.sdxl:
        tosave = {"model_0": MoFu.to(dtp(cfg.save_precision))}
    else:
        tosave = {"model_0": MoFu[0].to(dtp(cfg.save_precision)), "model_1": MoFu[1].to(dtp(cfg.save_precision))}
    pathto = Path(cfg.output_dir + cfg.MoFU_name + ".safetensors")
    
    save_file(tosave, pathto, {"name": cfg.MoFU_name, "base_model": cfg.diffusers_name, "version": '2', "notes": cfg.notes})

def main(cfg: MoFUConfig):
    if cfg.version == 1:
        dset = Dataset(cfg.dataset_dir)
        tags = get_useful_tags(dset, cfg)
        create_MoFU(tags, cfg)
    if cfg.version == 2:
        dset = SFDataset(cfg.artist_file)
        tags = dset.get_tags()
        create_MoFUv2(tags, cfg)
        
        
if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', "-c", type=str, help='config file path')
    args = parse.parse_args()
    if args.config:
        config = MoFUConfig(args.config)
        main(config)
    else:
        main(MoFUConfig())