How to run training for Pong games with DQN.
1) pip install -r requirements.txt
2) run notebook for training: when_should_agents_explore-TRAINING.ipynb
3) run notebook for game visualization: when_should_agents_explore-REPLAY.ipynb

To show training progress run tensorboard from logdir:

tensoboard --logdir /data/zabolotny-av/RL/logs --port 8991 --bind_all

EXAMPLE OF TRAINED MODEL:

![game_vis](https://github.com/zaaabik/RL/blob/main/model_130_game_vis.gif?raw=true)
