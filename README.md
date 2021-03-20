# Dual-phased Reinforcement Learning to Solve Pair-wise Routing for Express System
Zefang Zong, Yikang Zhou,Hengliang Luo, Depeng Jin, Yong Li
Department of Electronic Engineering Tsinghua University, Beijing, China
Meituan-Dianping Group, Beijing, China
liyong07@tsinghua.edu.cn


DALPR is a novel structure to solve pair-wise routing problem in on-demand delivery systems.
## Train
Train a model for a 50 points problem with
```bash
python mn_TW_ml_opt.py ----num_training_points 50 ----num_test_points 50 --num_paths 5 --num_paths_to_ruin 2
```
## Test
A test process is much like a training process for this iteration improve system. To load a pretrained model, add arg ```bash --model_to_restore ```to show its path. 

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
