# mistake-in-retro-contest-of-OpenAI

My nickname is mistake in retro contest.
This is my source code of contest, and this maybe a bit messy, I'm sorry for it.
I will reorgnanize it later.

I only try model of rainbow during the contest, so there is no PPO or jerk model in my code. Maybe I will add them later.

In order to adjust some parameters, some classes and functions in the project are copied from the anyrl-py(https://github.com/unixpickle/anyrl-py)

#about transfer learning

During the contest period, We tried the following two approaches to train the
model of rainbow:

1. Design the Network only, then submit the job that parameters are not pre-train,

2. Design the Network and per-train the parameters locally (about 7 millions timesteps), then submit the job.

We choose the first way to train the model. It's score is about 2000 more than
the second one. The reason might be that the second approach loses the
randomization of learning after per-training.

#some optimizations base the OpenAI-baselines

1. After testing the parameter named Nstep, we found the best value is 4, we think it's a balance point between the current and the future.

2. The model using nature_cnn training 1 million steps costs about 9 hours,while the max limit time is 12 hours, which leads to an oppotunity to design a complex network to replace the nature_cnn. After testing a lot of models, we found the network with one layer of cnn added on nature_cnn is the best.

3. The parameter of target_interval which the baseline uses is 8192. We found it takes too long between two *replace* operations due to the near convergence of the training process after 1000 steps, so We choose 1024 as the final value of target_interval.
