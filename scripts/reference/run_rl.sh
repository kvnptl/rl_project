#!/bin/bash

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python "$script_dir/agent_training.py" --total-timesteps 5000000 --learning-rate 3e-4 --buffer-size 3000000 --batch-size 1024 --n-envs 20 --parallel --max-steps 500 --train-freq 5 --gradient-steps 10 --learning-starts 20000 --tensorboard --checkpoint-freq 500000 --eval-freq 500000 --observation-type multimodal
