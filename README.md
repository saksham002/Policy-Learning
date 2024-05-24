# Environment Setup

## For Linux

```bash
# Instruction to set up on Linux

cd A2
conda env create -f environment_lin.yml
conda activate col864_a2
pip install -e .
```

## For Windows/macOS

```bash
# Environment setup for Win/Mac users. (Not tested rigorously)
cd A2
conda create -n col864_a2 python=3.9
conda activate col864_a2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt
pip install -e .
```

# Training

The training can be initiated from the `A2` folder as follows:

```bash
python scripts/train_agent.py --env_name <ENV> --exp_name <ALGO> [optional tags]
```

| Tag                | Description                                                                     | Possible values                    |
|--------------------|---------------------------------------------------------------------------------|------------------------------------|
| `env_name`         | The name of the MuJoCo Environment                                              | `Hopper-v4`, `HalfCheetah-v4`, `Ant-v4` |
| `exp_name`         | The algorithm you want to use to train the policy                               | `imitation`, `RL`, `imitation-RL`  |
| `no_gpu`           | Include this tag if you want to train on CPU. This assignment is designed to be feasible on a CPU. Therefore, having a GPU is not mandatory |                                    |
| `scalar_log_freq`  | The frequency with which you want to log your metrics like loss and rewards     | positive integer                   |
| `video_log_freq`   | The frequency with which you want to log the visual performance of your model   | positive integer                   |
| `load_checkpoint`  | You can load a checkpoint model to resume training. However, your final submission will undergo uninterrupted training on our servers. | `<path to the checkpoint>`         |

