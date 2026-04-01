from __future__ import annotations

from typing import Any, Dict, Sequence
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from model import PolicyAwareHeuristicNet
from reward_function_new import MSARewardFunction
from cordon_environment import CordonEnv
from deepaco import DeepACOAgent
from data_generation import generate_dataset, split_dataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env_factory(instance: Dict[str, Any], max_steps: int | None = None):
    road_graph = instance["road_graph"]
    virtual_node = int(instance.get("virtual_node", -1))
    if max_steps is None:
        max_steps = road_graph.number_of_nodes() + 2

    def _factory():
        return CordonEnv(
            road_graph=road_graph,
            max_steps=max_steps,
            virtual_node=virtual_node,
        )

    return _factory


def build_reward_fn(
    instance: Dict[str, Any],
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
):
    reward_fn = MSARewardFunction(
        links=instance["links"],
        od_demand=instance["od_demand"],
        virtual_node=instance.get("virtual_node", -1),
        reward_weights=reward_weights,
        solver_kwargs=solver_kwargs,# 
    )
    reward_fn.compute_initial(
        zone_nodes_or_state=None,
        policy_kind="none",
        policy_value=None,
    )
    return reward_fn


def sum_rollout_logprob(rollout, device):
    if len(rollout.log_probs) == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    ### 这里已经改成了mean
    return torch.stack(rollout.log_probs).mean()


def mean_rollout_entropy(rollout, device):
    if len(rollout.entropies) == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    ###
    return torch.stack(rollout.entropies).mean()


def replay_rollout_metrics(
    agent: DeepACOAgent,
    env_factory,
    heu_new: torch.Tensor,
    heu_ref: torch.Tensor,
    rollout,
):
    env = env_factory()
    env.reset()

    step_logps = []
    step_ents = []
    step_kls = []

    for action in rollout.actions:
        actions_new, probs_new, dist_new = agent._action_distribution(env, heu_new)
        actions_ref, probs_ref, dist_ref = agent._action_distribution(env, heu_ref)
        if actions_new != actions_ref:
            raise RuntimeError(
                f"Action support mismatch.\nnew={actions_new}\nref={actions_ref}"
            )
        action = int(action)
        if action not in actions_new:
            raise RuntimeError(f"Replay action {action} not found in legal actions {actions_new}")

        idx = actions_new.index(action)
        idx_t = torch.tensor(idx, dtype=torch.long, device=heu_new.device)
        logp = dist_new.log_prob(idx_t)
        ent = dist_new.entropy()
        kl = kl_divergence(dist_new, dist_ref)

        step_logps.append(logp)
        step_ents.append(ent)
        step_kls.append(kl)

        out = env.step(action)
        if out.done or action == agent.virtual_node:
            break
    if len(step_logps) == 0:
        zero = torch.zeros((), dtype=torch.float32, device=heu_new.device)
        return zero, zero, zero
    mean_logp = torch.stack(step_logps).mean()
    mean_ent = torch.stack(step_ents).mean()
    mean_kl = torch.stack(step_kls).mean()
    return mean_logp, mean_ent, mean_kl

def replay_rollout_logprob(
    agent: DeepACOAgent,
    env_factory,
    heu_mat: torch.Tensor,
    rollout,
):

    env = env_factory()
    env.reset()

    log_probs = []
    entropies = []

    for action in rollout.actions:
        lp, ent = agent.evaluate_action_log_prob(env, heu_mat, action)
        log_probs.append(lp)
        entropies.append(ent)
        out = env.step(action)
        if out.done or action == agent.virtual_node:
            break

    if len(log_probs) == 0:
        zero = torch.zeros((), dtype=torch.float32, device=heu_mat.device)
        return zero, zero
    return torch.stack(log_probs).mean(), torch.stack(entropies).mean()
''''''

# =========================================================
# validation
# =========================================================
@torch.no_grad()
def validate_dataset_quality(
    dataset: Sequence[Dict[str, Any]],
    model: nn.Module | None = None,
    device: str = "cpu",
):
    """
    Check whether the training set itself is well-formed.
    """
    report = {
        "num_instances": len(dataset),
        "num_valid": 0,
        "num_invalid": 0,
        "mean_num_nodes": 0.0,
        "mean_num_od": 0.0,
        "forward_ok": 0,
        "reward_ok": 0,
        "errors": [],
    }

    if model is None:
        model = PolicyAwareHeuristicNet(hidden_dim=64).to(device)
    model.eval()

    total_nodes = 0
    total_od = 0

    for idx, inst in enumerate(dataset):
        try:
            data = inst["pyg_data"].to(device)
            total_nodes += data.num_nodes
            total_od += len(inst["od_demand"])

            # graph-data shape checks
            assert data.road_x.size(0) == data.od_x.size(0)
            assert data.pair_feat.size(0) == data.pair_feat.size(1) == data.num_nodes
            assert inst["road_graph"].number_of_nodes() + 1 == data.num_nodes

            # model forward check
            heur = model(data)
            assert heur.shape == (data.num_nodes, data.num_nodes)
            assert torch.isfinite(heur).all()
            report["forward_ok"] += 1

            # reward check
            reward_fn = build_reward_fn(inst)
            out = reward_fn.evaluate(
                zone_nodes_or_state=None, ####
                policy_kind=inst["policy_type"],
                policy_value=inst["policy_value"],
            )
            assert np.isfinite(out["reward"])
            report["reward_ok"] += 1

            report["num_valid"] += 1

        except Exception as e:
            report["num_invalid"] += 1
            report["errors"].append((idx, str(e)))

    if len(dataset) > 0:
        report["mean_num_nodes"] = total_nodes / len(dataset)
        report["mean_num_od"] = total_od / len(dataset)

    return report


@torch.no_grad()
def evaluate_on_validation(
    model: nn.Module,
    val_set: Sequence[Dict[str, Any]],
    device: str = "cpu",
    n_ants: int = 16,
    n_rounds: int = 3,
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
):
    """
    Validation score = average best reward across validation instances.
    """
    model.eval()

    rewards = []
    paths = []

    for inst in val_set:
        data = inst["pyg_data"].to(device)
        heu_mat = model(data)

        reward_fn = build_reward_fn(
            inst,
            reward_weights=reward_weights,
            solver_kwargs=solver_kwargs,
        )

        agent = DeepACOAgent(
            node2idx=data.node2idx,
            n_ants=n_ants,
            alpha=1.0,
            beta=1.0,
            decay=0.9,
            max_steps=inst["road_graph"].number_of_nodes() + 2,
            virtual_node=inst.get("virtual_node", -1),
            elitist=False,
            min_max=False,
            device=device,
        )

        env_factory = make_env_factory(inst)

        out = agent.run(
            env_factory=env_factory,
            heu_mat=heu_mat,
            reward_fn=reward_fn,
            n_rounds=n_rounds,
            inference=False,
            update_pheromone=True,
            terminal_reward_only=True,
            policy_kind=inst["policy_type"],
            policy_value=inst["policy_value"],
        )
        rewards.append(float(out["best_reward"]))
        paths.append(out["best_path"])

    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
    return {
        "mean_best_reward": mean_reward,
        "all_best_rewards": rewards,
        "all_best_paths": paths,
    }


# =========================================================
# reinforce
# =========================================================
def reinforce_train(
    train_set: Sequence[Dict[str, Any]],
    val_set: Sequence[Dict[str, Any]],
    device: str = "cpu",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_ants: int = 16,
    entropy_coef: float = 1e-3,
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
    checkpoint_path: str = "reinforce_best.pt",):

    model = PolicyAwareHeuristicNet(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0

        for inst in train_set:
            data = inst["pyg_data"].to(device)
            heu_mat = model(data)

            reward_fn = build_reward_fn(
                inst,
                reward_weights=reward_weights,
                solver_kwargs=solver_kwargs,
            )

            agent = DeepACOAgent(
                node2idx=data.node2idx,
                n_ants=n_ants,
                alpha=1.0,
                beta=1.0,
                decay=0.9,
                max_steps=inst["road_graph"].number_of_nodes() + 2,
                virtual_node=inst.get("virtual_node", -1),
                elitist=False,
                min_max=False,
                device=device,
            )

            env_factory = make_env_factory(inst)

            run_out = agent.run(
                env_factory=env_factory,
                heu_mat=heu_mat,
                reward_fn=reward_fn,
                n_rounds=1,
                inference=False,
                update_pheromone=False,   # train阶段先不做pheromone replay
                terminal_reward_only=True,
                policy_kind=inst["policy_type"],
                policy_value=inst["policy_value"],
            )

            rollouts = run_out["last_rollouts"]
            rewards = torch.tensor(
                [float(r.total_reward) for r in rollouts],
                dtype=torch.float32,
                device=device,
            )
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            traj_logps = torch.stack([sum_rollout_logprob(r, device) for r in rollouts])
            traj_ents = torch.stack([mean_rollout_entropy(r, device) for r in rollouts])

            loss_pg = -(adv.detach() * traj_logps).mean()
            loss_ent = -entropy_coef * traj_ents.mean()
            loss = loss_pg + loss_ent

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_reward += float(rewards.mean().item())

        val_out = evaluate_on_validation(
            model=model,
            val_set=val_set,
            device=device,
            n_ants=n_ants,
            n_rounds=3,
            reward_weights=reward_weights,
            solver_kwargs=solver_kwargs,
        )

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_set), 1),
            "train_reward": epoch_reward / max(len(train_set), 1),
            "val_reward": val_out["mean_best_reward"],
        }
        history.append(record)
        print(record)

        if record["val_reward"] > best_val:
            best_val = record["val_reward"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "history": history,
                    "best_val_reward": best_val,
                },
                checkpoint_path,
            )

    return model, history


# =========================================================
# grpo
# =========================================================
'''
def grpo_train(
    train_set: Sequence[Dict[str, Any]],
    val_set: Sequence[Dict[str, Any]],
    device: str = "cpu",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_ants: int = 16,
    clip_eps: float = 0.2,
    grpo_updates: int = 3,
    entropy_coef: float = 1e-3,
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
    checkpoint_path: str = "grpo_best.pt",
):
    model = PolicyAwareHeuristicNet(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0

        for inst in train_set:
            data = inst["pyg_data"].to(device)

            reward_fn = build_reward_fn(
                inst,
                reward_weights=reward_weights,
                solver_kwargs=solver_kwargs,
            )

            agent = DeepACOAgent(
                node2idx=data.node2idx,
                n_ants=n_ants,
                alpha=1.0,
                beta=1.0,
                decay=0.9,
                max_steps=inst["road_graph"].number_of_nodes() + 2,
                virtual_node=inst.get("virtual_node", -1),
                elitist=False,
                min_max=False,
                device=device,
            )

            env_factory = make_env_factory(inst)

            # collect one group with current policy
            with torch.no_grad():
                heu_old = model(data)
                run_out = agent.run(
                    env_factory=env_factory,
                    heu_mat=heu_old,
                    reward_fn=reward_fn,
                    n_rounds=1,
                    inference=False,
                    update_pheromone=False,
                    terminal_reward_only=True,
                    policy_kind=inst["policy_type"],
                    policy_value=inst["policy_value"],
                )
                rollouts = run_out["last_rollouts"]

            rewards = torch.tensor(
                [float(r.total_reward) for r in rollouts],
                dtype=torch.float32,
                device=device,
            )
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            old_logp = torch.stack([sum_rollout_logprob(r, device) for r in rollouts]).detach()
            # 长度惩罚?
            # several clipped updates on the same group
            for _ in range(grpo_updates):
                heu_new = model(data)

                new_logp = []
                new_ent = []
                for r in rollouts:
                    # 这里的log ---log(P/\sum(P))
                    lp, ent = replay_rollout_logprob(agent, env_factory, heu_new, r)
                    new_logp.append(lp)
                    new_ent.append(ent)

                new_logp = torch.stack(new_logp)
                new_ent = torch.stack(new_ent)

                ratio = torch.exp(new_logp - old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                obj1 = ratio * adv.detach()
                obj2 = clipped_ratio * adv.detach()
                loss_pg = -torch.min(obj1, obj2).mean()
                loss_ent = -entropy_coef * new_ent.mean() # KL 散度去哪了
                loss = loss_pg + loss_ent

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                epoch_loss += float(loss.item())

            epoch_reward += float(rewards.mean().item())

        val_out = evaluate_on_validation(
            model=model,
            val_set=val_set,
            device=device,
            n_ants=n_ants,
            n_rounds=3,
            reward_weights=reward_weights,
            solver_kwargs=solver_kwargs,
        )

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_set), 1),
            "train_reward": epoch_reward / max(len(train_set), 1),
            "val_reward": val_out["mean_best_reward"],
        }
        history.append(record)
        print(record)

        if record["val_reward"] > best_val:
            best_val = record["val_reward"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "history": history,
                    "best_val_reward": best_val,
                },
                checkpoint_path,
            )

    return model, history
'''



def grpo_train(
    train_set: Sequence[Dict[str, Any]],
    val_set: Sequence[Dict[str, Any]],
    device: str = "cpu",
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_ants: int = 16,
    clip_eps: float = 0.2,
    grpo_updates: int = 3,
    entropy_coef: float = 1e-3,
    beta_kl: float = 0.02,
    reward_weights: Dict[str, float] | None = None,
    solver_kwargs: Dict[str, Any] | None = None,
    checkpoint_path: str = "grpo_best.pt",
):
    """
    GRPO training for your ACO-based constructive policy.

    Key points:
    - collect one group of rollouts with current model
    - rewards are group-normalized to build advantages
    - use PPO-style clipped ratio objective
    - add exact KL( pi_new || pi_ref ) regularization
    - use mean log-prob instead of sum log-prob
    """

    model = PolicyAwareHeuristicNet(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    best_val = float("-inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0

        for inst in train_set:
            data = inst["pyg_data"].to(device)

            reward_fn = build_reward_fn(
                inst,
                reward_weights=reward_weights,
                solver_kwargs=solver_kwargs,
            )

            agent = DeepACOAgent(
                node2idx=data.node2idx,
                n_ants=n_ants,
                alpha=1.0,
                beta=1.0,
                decay=0.9,
                max_steps=inst["road_graph"].number_of_nodes() + 2,
                virtual_node=inst.get("virtual_node", -1),
                elitist=False,
                min_max=False,
                device=device,
            )

            env_factory = make_env_factory(inst)

            # -------------------------------------------------
            # Step 1: collect one rollout group with current policy
            # -------------------------------------------------
            with torch.no_grad():
                heu_old = model(data)
                run_out = agent.run(
                    env_factory=env_factory,
                    heu_mat=heu_old, # heu_old从哪来
                    reward_fn=reward_fn,
                    n_rounds=1,
                    inference=False,
                    update_pheromone=False,
                    terminal_reward_only=True,
                    policy_kind=inst["policy_type"],
                    policy_value=inst["policy_value"],
                )
                rollouts = run_out["last_rollouts"]

            rewards = torch.tensor(
                [float(r.total_reward) for r in rollouts],
                dtype=torch.float32,
                device=device,
            )

            # group-relative normalized advantage
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            # old policy log-prob from sampled rollouts
            old_logp = torch.stack(
                [sum_rollout_logprob(r, device) for r in rollouts]
            ).detach()

            # frozen reference heuristic
            with torch.no_grad():
                heu_ref = ref_model(data)

            # -------------------------------------------------
            # Step 2: multiple clipped updates on the same group
            # -------------------------------------------------
            for _ in range(grpo_updates):
                heu_new = model(data)

                new_logp = []
                new_ent = []
                new_kl = []

                for r in rollouts:
                    lp, ent, kl = replay_rollout_metrics(
                        agent=agent,
                        env_factory=env_factory,
                        heu_new=heu_new,
                        heu_ref=heu_ref,
                        rollout=r,
                    )
                    new_logp.append(lp)
                    new_ent.append(ent)
                    new_kl.append(kl)

                new_logp = torch.stack(new_logp)
                new_ent = torch.stack(new_ent)
                new_kl = torch.stack(new_kl)

                ratio = torch.exp(new_logp - old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

                obj1 = ratio * adv.detach()
                obj2 = clipped_ratio * adv.detach()

                loss_pg = -torch.min(obj1, obj2).mean()
                loss_kl = beta_kl * new_kl.mean()
                loss_ent = -entropy_coef * new_ent.mean()

                loss = loss_pg + loss_kl + loss_ent

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()

                epoch_loss += float(loss.item())

            epoch_reward += float(rewards.mean().item())

        # -------------------------------------------------
        # validation
        # -------------------------------------------------
        val_out = evaluate_on_validation(
            model=model,
            val_set=val_set,
            device=device,
            n_ants=n_ants,
            n_rounds=3,
            reward_weights=reward_weights,
            solver_kwargs=solver_kwargs,
        )

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_set), 1),
            "train_reward": epoch_reward / max(len(train_set), 1),
            "val_reward": val_out["mean_best_reward"],
        }
        history.append(record)
        print(record)

        if record["val_reward"] > best_val:
            best_val = record["val_reward"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": hidden_dim,
                    "history": history,
                    "best_val_reward": best_val,
                },
                checkpoint_path,
            )

    return model, history

# =========================================================
# example main
# =========================================================
if __name__ == "__main__":
    set_seed(42)
    # 应该制定
    dataset = generate_dataset(
        num_instances=40,
        grid_shapes=((3, 3), (3, 4), (4, 4)),
        seed=42,
        virtual_node=-1,
    )
    train_set, val_set = split_dataset(dataset, val_ratio=0.2, seed=42)

    quality_report = validate_dataset_quality(train_set, device="cpu")
    print("dataset quality:", quality_report)
    # 
    model, hist = reinforce_train(
        train_set=train_set,
        val_set=val_set,
        device="cpu",
        epochs=5,
        lr=1e-3,
        hidden_dim=64,
        n_ants=8,
        checkpoint_path="reinforce_best.pt",
    )

    # 如果要训练 GRPO，则改成：
    # model, hist = grpo_train(
    #     train_set=train_set,
    #     val_set=val_set,
    #     device="cpu",
    #     epochs=5,
    #     lr=1e-3,
    #     hidden_dim=64,
    #     n_ants=8,
    #     checkpoint_path="grpo_best.pt",
    # )