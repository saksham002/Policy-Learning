# Description

This code provides the implementation of Imitation Learning, Reinforcement Learning (RL) and Imitation-Seeded RL for MuJoCo (Multi-Joint dynamics with Contact) environments from the MuJoCo Gym.


# Environment Setup

## For Linux

```bash
# Instruction to set up on Linux

conda env create -f environment_lin.yml
conda activate policy-learning
pip install -e .
```

## For Windows/macOS

```bash
# Environment setup for Win/Mac users.

conda create -n policy-learning python=3.9
conda activate policy-learning
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt
pip install -e .
```

# Training

The training can be initiated as follows:

```bash
python scripts/train_agent.py --env_name <ENV> --exp_name <ALGO> [optional tags]
```

| Tag                | Description                                                                     | Possible values                    |
|--------------------|---------------------------------------------------------------------------------|------------------------------------|
| `env_name`         | The name of the MuJoCo Environment.                                              | `Hopper-v4`, `HalfCheetah-v4`, `Ant-v4` |
| `exp_name`         | The algorithm to be used to train the policy.                               | `imitation`, `RL`, `imitation-RL`  |
| `no_gpu`           | Include this tag to train on CPU. |                                    |
| `scalar_log_freq`  | The frequency with which to log metrics like loss and rewards.     | positive integer                   |
| `video_log_freq`   | The frequency with which to log the visual performance of the model.   | positive integer                   |
| `load_checkpoint`  | A checkpoint can be loaded to resume training. | `<path to the checkpoint>`         |

The configs and hyper-parameters used for model creation and training are defined in `config.py`. The best model (based on cumulative rewards from simulated trajectories) from one training execution for a set environment and algorithm will be saved with the name "model_<env_name>_<exp_name>.pth" inside the directory "best_models".

# Visualization

TensorBoard can be used for visualization purposes. The TensorBoard directories will be created inside the directory "data". On a local machine, the saved simulations/plots can be viewed by running:

```bash
tensorboard --logdir data/tensorboard_dir
```
If training is done on a remote server, port forwarding needs to be used to view the tensor board in the local browser.
```bash
tensorboard --logdir <path> --port 6006
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>

# Now open http://localhost:16006
```

