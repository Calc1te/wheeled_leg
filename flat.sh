source ~/IsaacLab/env_isaaclab/bin/activate

python3 /home/trent/wheeled_leg/wheeled_leg/scripts/rsl_rl/train.py --task frog-flat-v0 --headless --num_envs ${1:-16}