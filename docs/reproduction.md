# Reproducing the robotic pushing experiments

This project compares observation inputs and reward functions for a Franka Panda trained with Soft Actor-Critic in PyBullet. We replayed four released checkpoints under a shared protocol. Both multimodal policies reached the goal, while the two single-modality policies timed out in all three trials. [Archived experiments](../results/)

## Task and method

A frontal camera provides simulator segmentation masks. The policy receives the object's normalized image bounding box, optional target coordinates, and optional Cartesian end-effector position and velocity. During occlusion the environment retains the last detected box. The SAC actor outputs a planar displacement. The simulator applies smoothing, inverse kinematics, and joint position control at a fixed pushing height. One policy action advances ten simulation steps.

The five-level curriculum widens the spawn range around the tabletop. The reference trainer's progression rule uses the trailing 50 episodes and an 80% success threshold. The shaped reward combines reaching the goal, approaching the object, moving it toward the goal, and penalties for elapsed time and occlusion. The sparse variant retains goal completion and a time penalty. The baseline uses a fixed target, so a policy can learn a useful pushing direction without receiving explicit target coordinates. [Reference trainer](../scripts/reference/agent_training.py)

## Archived training comparisons

The archived experiments measure the fraction of episodes where the object reaches the target. Both multimodal runs reached 100% on the logged metric, while the vision-only and proprioception-only runs peaked at 8% and 51%, respectively. The goal-conditioned run first logged 100% at about 0.99 million steps and the multimodal run without goal input did so at about 1.15 million steps. The sparse-reward run reached that mark at about 1.13 million steps. These are first recorded hits in individual training curves. [Raw metric exports](../results/)

![Archived training curves comparing observation modes](../results/plots/step_vs_success_rate_plot.png)

- [Goal-conditioning curves](../results/plots/step_vs_success_rate_plot_multimodal_compare.png)
- [Shaped and sparse reward curves](../results/plots/step_vs_success_rate_plot_reward_dense_vs_sparse.png)
- [Raw metric exports](../results/)

## Checkpoint replay

We evaluated the four checkpoint filenames from the archived sweep on CPU with Python 3.11 and the pinned dependencies. Each checkpoint used deterministic actions, three hard-difficulty episodes, seed 42, and a 150-step cap. [Experiment sweep](../scripts/launchers/run_rl_eval_sweep.sh)

| Observation mode | Goals reached | Mean steps | Mean reward |
| --- | ---: | ---: | ---: |
| Vision and goal | 0/3 | 150.00 | −100.58 |
| Proprioception | 0/3 | 150.00 | −60.96 |
| Vision, proprioception, and goal | 2/3 | 72.33 | 147.54 |
| Vision and proprioception | 1/3 | 109.67 | 31.00 |

The multimodal policy with goal input reached the target in 38 and 29 steps on its successful trials. Both single-modality policies hit the 150-step cap on every trial. The result agrees with the direction of the archived experiments. Three trials per policy give a limited estimate of success. [Episode-level results](../results/replay_observation_modes.csv)

Run the following from the repository root after installing the dependencies and object models. Set `mode` and `checkpoint` for the desired run. [Setup instructions](../README.md#run-it-locally)

```bash
mode=multimodal
checkpoint=checkpoints/models_exp_05_multimodal/interrupted_model_sac_20250327_083255.zip
python scripts/stage_6_train_agent_exp_03_eval.py \
  --eval-only --trust-checkpoint --load-model "$checkpoint" \
  --observation-type "$mode" --difficulty-level 3 --seed 42 \
  --max-steps 150 --n-eval-episodes 3
```

| Mode | Checkpoint path |
| --- | --- |
| Vision and goal | `checkpoints/models_exp_03_vision_only/final_model_sac_20250326_031046.zip` |
| Proprioception | `checkpoints/models_exp_04_proprio_only/final_model_sac_20250326_175631.zip` |
| Vision, proprioception, and goal | `checkpoints/models_exp_05_multimodal/interrupted_model_sac_20250327_083255.zip` |
| Vision and proprioception | `checkpoints/models_exp_06_multimodal_no_goal/interrupted_model_sac_20250327_141302.zip` |

## Recorded demonstration

The video uses the experiment 05 multimodal checkpoint and seed 36 with a sampled hard-difficulty spawn. The episode reached the goal in 27 actions. The demonstration uses a selected starting position. The same checkpoint reached the 150-step cap on the recorder's preset hard spawn. [Video](../results/videos/multimodal-hard-replay.mp4) · [Recording command](../README.md#record-a-policy)

## Limits

A fresh learning-curve reproduction would need the exact settings and stopping rule for each run, multiple independent seeds, and a stated convergence criterion. This replay covers a single object and a fixed target in simulation. The reward and varying-target checkpoints need their own replay protocols. Transfer to physical hardware remains future work.
