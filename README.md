This repository show research in direction of exploration/exploitation in Reinforsment learning.
In our project we apply 4 blind ε-greedy strategy of switching between exploration/exploitation:

* Only explore strategy ε = 1
* Only policy ε = 0
* Epsilon greedy ε = 1
* Epsilon greedy with exponential decay 


Epsilon greedy with exponential decay plot.


Experiments using Atari learning environment for three games: Pong, BeamRider and Breakout.

Pong results:

![pong](https://raw.githubusercontent.com/zaaabik/RL/main/assets/reward_different_strategy_pong.jpeg)

BeamRider results:

![beamrider](https://raw.githubusercontent.com/zaaabik/RL/main/assets/reward_different_strategy_beam_rider.jpeg)

Breakout results:

![breakout](https://raw.githubusercontent.com/zaaabik/RL/main/assets/reward_different_strategy_break_out.jpeg)


Results summary:
1) For pong game, we
find that only explore strategy does not work at all in case,
while e − greedy strategies show the best result.
2)  Beamrider game shows
the best results when we use the only-policy strategy of
exploitation, and all other methods get approximate equal
results.
3) Breakout game becomes hard for our model
stuck in a situation where a player should press the special
button after losing the life

-------------------------------------------------------
How to run training DQN for atari game.  Does not work for windows :((((
1) pip install -r requirements.txt
2) Choose the game: BeamRiderNoFrameskip-v4, PongNoFrameskip-v4, BreakoutNoFrameskip-v4
3) Choose exploration/explotation strategy: eps_greedy, eps_greedy_decay, only_explore, only_strategy
4) python experiments.py --device 'cuda:1' --mode eps_greedy --game_name BeamRiderNoFrameskip-v4

To show training progress run tensorboard from logdir:

tensoboard --logdir logs/when_should_agents_explore/ --port 8991 --bind_all

EXAMPLE OF TRAINED MODEL FOR PONG GAME (Model is in the right part):

![game_vis](https://github.com/zaaabik/RL/blob/main/model_130_game_vis.gif?raw=true)
