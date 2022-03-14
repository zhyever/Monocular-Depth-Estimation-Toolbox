import torch
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Process mono3d results.')
    parser.add_argument('--file-path', type=str, help='input file path')
    parser.add_argument('--out-path', type=str, help='output file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    new_checkpoint = {}
    new_checkpoint['state_dict'] = {}

    checkpoint = torch.load(args.file_path)
    params = checkpoint['state_dict']
    for key, val in params.items():
        if 'img_backbone' in key:
            new_key = key[4:]
            new_checkpoint['state_dict'][new_key] = val
    
    torch.save(new_checkpoint, args.out_path)

