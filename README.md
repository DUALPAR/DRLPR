# Dual-phased Reinforcement Learning to Solve Pair-wise Routing for Express System
Zefang Zong, Yikang Zhou,Hengliang Luo, Depeng Jin, Yong Li. Dual-phased Reinforcement Learning to Solve Pair-wise Routing for Express System


## Introduction

Dual-phased Reinforcement Learning for Pair-wise Routing(DALPR) is a novel structure to solve pair-wise routing problem in on-demand delivery systems.

## Dependencies

Python>=3.6

TensorFlow>=1.8

## Train DALPR

Train a model for a 50 points problem with
```bash
python mn_TW_ml_opt.py ----num_training_points 50 ----num_test_points 50 --num_paths 5 --num_paths_to_ruin 2
```

## Test

A test process is much like a training process for this iteration improve system. To load a pretrained model, add arg ```--model_to_restore ```to show its path. 

### GPU Configurations

GPU configurations are setted in '''mn_TW_ml_opt.py'''. By default, only GPU0 will be used. To use multiple GPUs, set the environment like this:
```python
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2"
```

## Acknowledgements

DRLPR is developed based on l2i(Hao Lu; Xingwen Zhang; Shuang Yang. 2020. A LEARNING-BASED ITERATIVE METHOD FOR SOLVING VEHICLE ROUTING PROBLEMS. In Proceedings of International Conference on Learning Representations (ICLR).) 

Thanks to their efforts and their repository is at github: https://github.com/rlopt/l2i

<!--
**DUALPAR/DUALPAR** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
