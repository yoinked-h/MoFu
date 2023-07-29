from model_handler import TokenizerModels
from dataloader import Dataset
from config import MoFUConfig #noqa
from pathlib import Path
from safetensors.torch import save_file
def get_useful_tags(dset: Dataset, cfg: MoFUConfig = MoFUConfig()):
    usage = dset.get_tags_and_usage()
    #filter out the less common tags
    percent = cfg.dropout_percentile
    datasize = len(dset)
    useful = []
    for tag in usage.keys():
        if usage[tag] > round(datasize * percent):
            useful.append(tag)
    return useful
def create_MoFU(tags, cfg: MoFUConfig = MoFUConfig()):
    model = TokenizerModels(cfg.diffusers_name, cfg.device)
    MoFU = model.encode(", ".join(tags))
    #save with safetensors
    tosave = {"model": MoFU}
    pathto = Path(cfg.output_dir + cfg.MoFU_name + ".safetensors")
    
    save_file(tosave, pathto, {"name": cfg.MoFU_name, "base_model": cfg.diffusers_name})

def main(cfg: MoFUConfig = MoFUConfig()):
    dset = Dataset(cfg.dataset_dir)
    tags = get_useful_tags(dset, cfg)
    create_MoFU(tags, cfg)

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('-c', '--config', help='config file path')
    args = parse.parse_args()
    if args.config:
        config = MoFUConfig(args.config)
        main(config)
    else:
        main(MoFUConfig())