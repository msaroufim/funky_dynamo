# Hard Dynamo examples
* [probtorch](probtorch.py): Probabilistic Learning
* [pyg](pyg.py): Geometric Deep Learning
* [stable-baselines.py](stable-baselines.py): Reinforcement Learning
* [pytorch3d.py](pytorch3d.py): Implement neural rendering on GPU

## Not supported

`torchdynamo` requires pytorch > 1.12 so lots of libraries aren't supported yet

* [MMF](https://github.com/facebookresearch/mmf) for multimodal learnings requires  1.6 < torch < 1.9   
* [PyG](https://www.pyg.org/) graph geometric library torch < 1.11
* [CrypTen](https://github.com/facebookresearch/CrypTen) an encrypted ML library, officially supported torch version is 1.9 but the launcher examples don't work for me when using 1.12
* [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/docs/examples/pulsar_basic.py): Strange issues when trying to import package, not a package error