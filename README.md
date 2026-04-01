# Cordon-DeepACO

## 项目简介 | Project Overview
- 中文：本仓库包含一个用于 **Cordon Design** 的 DeepACO 重构实现，核心代码位于 `DeepACO_reconstruct/`。
- EN: This repository contains a reconstructed DeepACO implementation for **cordon design**, with core code under `DeepACO_reconstruct/`.

## 代码结构 | Code Structure
- `DeepACO_reconstruct/model.py`  
  中文：策略感知启发式网络（PolicyAwareHeuristicNet），输出 ACO 使用的启发式矩阵。  
  EN: Policy-aware heuristic network that outputs the heuristic matrix for ACO.

- `DeepACO_reconstruct/deepaco.py`  
  中文：DeepACO 智能体，负责蚂蚁构造路径、动作采样、信息素更新与多轮搜索。  
  EN: DeepACO agent for path construction, action sampling, pheromone update, and multi-round search.

- `DeepACO_reconstruct/cordon_environment.py`  
  中文：带虚拟节点的 Cordon 环境定义，提供状态转移与可行动作集合。  
  EN: Cordon environment with a virtual node, including transitions and action constraints.

- `DeepACO_reconstruct/reward_function_new.py`  
  中文：MSA 下层分配与多目标 reward（效率/环境/可达性/公平）。  
  EN: MSA lower-level assignment and multi-objective reward (efficiency/environment/accessibility/equity).

- `DeepACO_reconstruct/train.py`  
  中文：训练入口，包含 Reinforce 与 GRPO 两类训练流程。  
  EN: Training entry with both Reinforce and GRPO pipelines.

- `DeepACO_reconstruct/deepaco_speed_limit_workflow.ipynb`  
  中文：可运行示例 Notebook（speed_limit-only），包含：
  1) 不训练随机初始化 + 10 蚂蚁 3 轮输出最优路径；
  2) Reinforce 训练 block；
  3) GRPO 训练 block。  
  EN: Runnable speed_limit-only demo notebook covering:
  1) untrained random-init heuristic + 10 ants / 3 rounds;
  2) Reinforce training block;
  3) GRPO training block.

## 运行方式 | How to Run
- 中文：建议在仓库根目录启动 Jupyter，然后打开 `DeepACO_reconstruct/deepaco_speed_limit_workflow.ipynb` 按顺序执行。  
- EN: Launch Jupyter from repo root, open `DeepACO_reconstruct/deepaco_speed_limit_workflow.ipynb`, and run cells top-to-bottom.

## 依赖提示 | Dependencies Note
- 中文：该重构实现依赖 `torch`, `torch_geometric`, `networkx`, `numpy` 等。请先准备对应环境。  
- EN: This reconstruction depends on `torch`, `torch_geometric`, `networkx`, `numpy`, etc. Please install them beforehand.
