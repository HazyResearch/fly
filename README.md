We use the template from `https://github.com/ashleve/lightning-hydra-template`.
Please read the instructions there to understand the repo structure.

## Implementation & Experiments

An example of Scatterbrain implementation (combining local attention and
Performer) is in the file `src/models/modules/attention/sblocal.py`.

### T2T-ViT inference on ImageNet
To run the T2T-ViT inference on ImageNet experiment:
1. Download the pretrained weights from the [T2T-ViT repo][https://github.com/yitu-opensource/T2T-ViT/releases]:
```sh
mkdir -p checkpoints/t2tvit
cd checkpoints/t2tvit
wget https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar
```
2. Convert the weights to the format compatible with our implementation of
   T2T-ViT:
```sh
# cd to scatterbrain path
python scripts/convert_checkpoint_t2t_vit.py checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar
```
3. Download the ImageNet dataset (just the validation set will suffice).
Below, `/path/to/imagenet` refers to the directory that contains the `train` and `val` directories.
4. Run the inference experiments:
```sh
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=full datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 81.7% acc
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=local datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 80.6% acc
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=performer datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 77.8-79.0% acc (there's randomness)
python run.py experiment=imagenet-t2tvit-eval.yaml model/t2tattn_cfg=sblocal datamodule.data_dir=/path/to/imagenet/ eval.ckpt=checkpoints/t2tvit/81.7_T2T_ViTt_14.pth.tar  # 81.1% acc
```

### MLP-Mixer-B with Pixelfly on ImageNet
With 8 GPUs, at least 32GB memory each:
```sh
python run.py experiment=imagenet/mixer/mixerb-cutmix-fbbflylr datamodule.data_dir=/path/to/imagenet model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.butterfly_size=8 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.n_factors=2 model.channel_mlp_cfg.linear1_cfg.sparse_cfg.sparsity_config.block=32 
```


## Requirements

Python 3.8+, Pytorch 1.9+, torchvision, torchtext, pytorch-fast-transformers, munch, einops, timm, hydra-core, hydra-colorlog, python-dotenv, rich, pytorch-lightning, lightning-bolts, triton.

We provide a Dockerfile that lists all the required packages.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{chen2021scatterbrain,
  title={Scatterbrain: Unifying Sparse and Low-rank Attention},
  author={Beidi Chen and Tri Dao and Eric Winsor and Zhao Song and Atri Rudra and Christopher R\'{e}},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
@article{chen2021pixelated,
  title={Pixelated Butterfly: Simple and Efficient Sparse training for Neural Network Models},
  author={Chen, Beidi and Dao, Tri and Liang, Kaizhao and Yang, Jiaming and Song, Zhao and Rudra, Atri and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations}
  year={2021}
}
@inproceedings{dao2022monarch,
  title={Monarch: Expressive structured matrices for efficient and accurate training},
  author={Dao, Tri and Chen, Beidi and Sohoni, Nimit S and Desai, Arjun and Poli, Michael and Grogan, Jessica and Liu, Alexander and Rao, Aniruddh and Rudra, Atri and R{\'e}, Christopher},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```
