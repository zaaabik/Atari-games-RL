How to run training DQN for atari game.
1) pip install -r requirements.txt
2) Choose the game: BeamRiderNoFrameskip-v4, PongNoFrameskip-v4, BreakoutNoFrameskip-v4
3) Choose exploration/explotation strategy: eps_greedy, eps_greedy_decay, only_explore, only_strategy
4) python experiments.py --device 'cuda:1' --mode eps_greedy --game_name BeamRiderNoFrameskip-v4

To show training progress run tensorboard from logdir:

tensoboard --logdir logs/when_should_agents_explore/ --port 8991 --bind_all

EXAMPLE OF TRAINED MODEL FOR PONG GAME (Model is in the right part):

![game_vis](https://github.com/zaaabik/RL/blob/main/model_130_game_vis.gif?raw=true)
