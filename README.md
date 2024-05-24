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
