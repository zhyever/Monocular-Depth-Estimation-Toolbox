
import torch

ckp = torch.load('nfs/saves/binsformer/binsformer_swinl_nyu.pth')


new_state_dict = {}

params = ckp['state_dict']
for key, val in params.items():
    if 'decode_head.transformer_decoder.decoder.' in key:
        new_key = 'decode_head.transformer_decoder.' + key.split('decode_head.transformer_decoder.decoder.')[1]
        new_state_dict[new_key] = val
    else:
        new_state_dict[key] = val


ckp['state_dict'] = new_state_dict

torch.save(ckp, 'nfs/saves/binsformer/binsformer_swinl_nyu_converted.pth')