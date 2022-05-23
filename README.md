How to run training for Pong games with DQN.
1) pip install -r requirements.txt
2) python experiments.py --device 'cuda:1' --mode eps_greedy --game_name BeamRiderNoFrameskip-v4

To show training progress run tensorboard from logdir:

tensoboard --logdir /data/zabolotny-av/RL/logs --port 8991 --bind_all

EXAMPLE OF TRAINED MODEL FOR PONG GAME (Model is in the right part):

![game_vis](https://github.com/zaaabik/RL/blob/main/model_130_game_vis.gif?raw=true)
