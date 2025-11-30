<div align="center">
  <img src="logo.png"  width="150px"  width="100" alt="EfficientFlow Logo"/>
  <h1>EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI</h1>
</div>

<div align="center">

[Jianlei Chang](https://chang-jl.github.io/)\*, [Ruofeng Mei](https://mrf2025.github.io/)\*, [Wei Ke](https://gr.xjtu.edu.cn/en/web/wei.ke/home/), [Xiangyu Xu](https://xuxy09.github.io/)†

Xi'an Jiaotong University

**AAAI 2026** &nbsp;|&nbsp; [Project Website](https://efficientflow.github.io) &nbsp;|&nbsp; [Paper](#)

</div>



## Installation
1. Hardware requirements

    It is recommended to use NVIDIA GeForce RTX 4090 GPU.
2. Install environment:

    ```
    conda env create -f conda_environment.yaml
    conda activate EfficientFlow
    ```



3. Install mimicgen:
    ```bash
    cd ..
    git clone https://github.com/NVlabs/mimicgen_environments.git
    cd mimicgen_environments
    pip install -e .
    ```

4. Update the source code: 

    Use `pip show robomimic` to identify the package installation path. Then, edit the file `robomimic/envs/env_robosuite.py` at `line 15` and replace the import statement:

    - **Original:** `import mimicgen_envs`
    - **Updated:** `import mimicgen`



## Dataset
### Download Dataset
Download dataset from [MimicGen](https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core).

Make sure the dataset is kept under `/path/to/EfficientFlow/data/robomimic/datasets/[dataset]/[dataset].hdf5`

### Convert Action Space in Dataset
The downloaded dataset has a relative action space. To train with absolute action space, the dataset needs to be converted accordingly
```bash
# Template
python EfficientFlow/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/[dataset]/[dataset].hdf5 -o data/robomimic/datasets/[dataset]/[dataset]_abs.hdf5 -n [n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., convert stack_d1 with 12 workers
python EfficientFlow/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/stack_d1/stack_d1.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 -n 12
```



## Training with image observation
To train EfficientFlow in Stack D1 task:
```bash 
python train.py --config-name=EfficientFlow task_name=stack_d1 n_demo=100
```
**Note**: Evaluation will be triggered automatically every certain number of epochs during training.


## Citation <a name="cite"></a>

### If you feel that this paper, models, or codes are helpful, please cite our paper, thanks for your support!

```bibtex
@inproceedings{chang2026EfficientFlow,
  author={Chang, Jianlei and Mei, Ruofeng and Ke, Wei and Xu, Xiangyu},
  title={EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```

## License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement
* Our code is built upon the origional [Equivarient Diffusion Policy](https://github.com/pointW/equidiff), [FlowPolicy](https://github.com/zql-kk/FlowPolicy), [MeanFlow](https://github.com/haidog-yaqub/MeanFlow), [MP1](https://github.com/LogSSim/MP1), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [ACT](https://github.com/tonyzhaozh/act), [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy).
