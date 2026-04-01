# Cordon-DeepACO

## 项目简介 | Project Overview
- This repository contains a reconstructed DeepACO implementation for **cordon design**, with core code under `DeepACO_reconstruct/`.

## 代码结构 | Code Structure
- `DeepACO_reconstruct/model.py`   
  Policy-aware heuristic network that outputs the heuristic matrix for ACO.

- `DeepACO_reconstruct/deepaco.py`  
  DeepACO agent for path construction, action sampling, pheromone update, and multi-round search.

- `DeepACO_reconstruct/cordon_environment.py`  
  Cordon environment with a virtual node, including transitions and action constraints.

- `DeepACO_reconstruct/reward_function_new.py`  
  MSA lower-level assignment and multi-objective reward (efficiency/environment/accessibility/equity).

- `DeepACO_reconstruct/train.py`  
  Training entry with both Reinforce and GRPO pipelines.

- `DeepACO_reconstruct/deepaco_speed_limit_workflow.ipynb`  
  Runnable speed_limit-only demo notebook covering:
  1) untrained random-init heuristic + 10 ants / 3 rounds;
  2) Reinforce training block;
  3) GRPO training block.

## How to Run
- Launch Jupyter from repo root, open `DeepACO_reconstruct/deepaco_speed_limit_workflow.ipynb`, and run cells top-to-bottom.

## Dependencies Note 
- This reconstruction depends on `torch`, `torch_geometric`, `networkx`, `numpy`, etc. Please install them beforehand.
