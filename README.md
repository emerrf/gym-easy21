# Gym Easy21

## Introduction

OpenAI Gym<sup>[2](#ref2)</sup> environment which reproduces the environment 
of RL Assignment Easy21 game with my solution.

The formulation of the assignment is in the file named `Easy21-Johannes.pdf`.
The lectures of the course are fortunatelly available on line:

Reinforcement Learning Course by David Silver<sup>[1](#ref1)</sup>.
Video Lectures:
- [Lecture 1: Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [Lecture 2: Markov Decision Process](https://www.youtube.com/watch?v=lfHX2hHRMVQ)
- [Lecture 3: Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4)
- [Lecture 4: Model-Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
- [Lecture 5: Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
- [Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps)
- [Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs)
- [Lecture 8: Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc)
- [Lecture 9: Exploration and Exploitation](https://www.youtube.com/watch?v=sGuiWX07sKw)
- [Lecture 10: Classic Games](https://www.youtube.com/watch?v=kZ_AUmFcZtk)

## Quick start
The code was written and tested with Python 2.7.12 on Ubuntu 16.04. 
The following instructions shows how to run it.
```bash
$ virtualenv easy21-devenv
$ source easy21-devenv/bin/activate
(easy21-devenv)$ pip install -r requirements.txt
(easy21-devenv)$ gym-easy21-run
```
Docker version available. Please be aware the plots will not render: 
```bash
$ docker build -t gym-easy21:1.0.0 .
$ docker run gym-easy21:1.0.0
```
For exploring inside the image:
```bash
$ docker run -it --entrypoint /bin/bash  gym-easy21:1.0.0
```
File `gym_easy21/envs/easy21_env.py` contains the environment implementation 
while `gym_easy21/envs/easy21_run.py` is my own solution of the Tasks. 

## Sample (random actions) 
The next snippet shows how to use the environment. The `easy21` variable 
contains the environment and the `reset` method initialise it. Then, using
 random `action`s, you can simulate a game with using the `step` method. 
 The `trace` variable captures the sequence of events in each iteration. `info`
 returns useful data for debugging. For more info please visit [Gym OpenAI 
 docs](https://gym.openai.com/docs).
 
```python
import gym
import gym_easy21
import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

easy21 = gym.make('Easy21-v0')

for i_episode in range(100):
    log.info("Episode {}".format(i_episode))
    trace = list()
    observation, _, done, info = easy21.reset()
    while not done:
        action = easy21.action_space.sample()
        observation, reward, done, info = easy21.step(action)
        trace.append((observation, action, reward))

    log.info(info)
    log.info(trace)

```
Output
```bash
[2017-06-18 20:44:08,165] Making new env: Easy21-v0
[2017-06-18 20:44:08,172] Episode 0
[2017-06-18 20:44:08,172] {'player_sum': 6, 'rewards': [-1], 'player_hand': [(6, 'B')], 'actions': ['stick'], 'dealer_sum': 19, 'dealer_hand': [(9, 'B'), (10, 'B')]}
[2017-06-18 20:44:08,172] [((9, 6), 0, -1)]
[2017-06-18 20:44:08,172] Episode 1
[2017-06-18 20:44:08,173] {'player_sum': -4, 'rewards': [0, -1], 'player_hand': [(1, 'B'), (2, 'B'), (7, 'R')], 'actions': ['hit', 'hit'], 'dealer_sum': 1, 'dealer_hand': [(1, 'B')]}
[2017-06-18 20:44:08,173] [((1, 3), 1, 0), ((1, -4), 1, -1)]
[2017-06-18 20:44:08,173] Episode 2
[2017-06-18 20:44:08,173] {'player_sum': 5, 'rewards': [1], 'player_hand': [(5, 'B')], 'actions': ['stick'], 'dealer_sum': -1, 'dealer_hand': [(6, 'B'), (3, 'B'), (3, 'R'), (8, 'B'), (8, 'R'), (7, 'R')]}
[2017-06-18 20:44:08,173] [((6, 5), 0, 1)]
...
```

## acknowledgement 
- I cannot thank enough David Silver for posting the course online. 
It is just brilliant and highly recommended to anyone who is interested in 
learning RL.
- Unfortunately, I did not attend the course so I found useful to compare my 
implementation with Matteo Hessel (mtths) solution also published on his 
[account](https://github.com/mtthss/easy21). Grazie mille! 

## References
<a name="ref1">[1]</a>: David Silver. [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) 

<a name="ref2">[2]</a>: Greg Brockman, Vicki Cheung, Ludwig Pettersson, 
Jonas Schneider, John Schulman, Jie Tang, Wojciech Zaremba. OpenAI Gym. 
June 2016. arXiv:1606.01540.