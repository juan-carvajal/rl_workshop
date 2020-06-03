from collections import deque
import random
from numpy import log as ln
from numpy import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import cos
from math import pi
import matplotlib.pyplot as plt
import os
import torch.autograd as autograd
from statistics import mean
import time

import time


class NN(nn.Module):

    def __init__(self, model):
        super(NN, self).__init__()
        # self.norm=nn.LayerNorm(model.state_dim)
        self.model=model


        self.fc = nn.Sequential(
            nn.Linear(model.state_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, model.action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class TrainingAgent:

    def __init__(self, model, runs, steps_per_run, batch_size, min_epsilon=0.05, gamma=0.99,
                 memory=5000, use_gpu=False, epsilon_min_percentage=0.1, epsilon_function='linear',lr=3e-4):

        if epsilon_function == 'linear':
            self.epsilon_function = self.linear_epsilon
        elif epsilon_function == 'log':
            self.epsilon_function = self.log_epsilon
        elif epsilon_function == 'constant':
            self.epsilon_function = self.constant_epsilon
        elif epsilon_function == 'cos':
            self.epsilon_function = self.cos_epsilon
        elif epsilon_function == 'gompertz':
            self.epsilon_function = self.gompertz_epsilon
        elif epsilon_function == 'consv2':
            self.epsilon_function = self.constant_v2_epsilon
        elif epsilon_function =='logv2':
            self.epsilon_function=self.logv2
        else:
            raise Exception(
                'The epsilon_function parameter must be one of these types: (linear, log, constant, cos , gompertz, consv2).')
        self.model = model
        self.memory = Memory(memory)
        self.epsilon = 1
        self.batch_size = batch_size
        self.epsilon_min_percentage = epsilon_min_percentage
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.runs = runs
        self.step_per_run = steps_per_run
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tensor = torch.cuda if self.use_gpu else torch
        self.model1 = NN(model)
        self.model2=NN(model)
        self.tau=0.01
        if self.use_gpu:
            print("Training model on GPU.")
            self.model1.to("cuda")
            self.model2.to("cuda")
        else:
            print("Training model on CPU.")

        self.loss1 = nn.SmoothL1Loss()
        self.loss2 = nn.SmoothL1Loss()
        self.optizer1 = optim.Adam(self.model1.parameters(),lr=lr, amsgrad=True,weight_decay=0.05)
        self.optizer2 = optim.Adam(self.model2.parameters(), lr=lr, amsgrad=True,weight_decay=0.05)

    def run(self):
        run_rewards = []
        time_sum = deque(maxlen=10)
        for run in range(self.runs):

            start_time = time.time()

            total_reward = 0

            epsilon = self.epsilon_function(run)
            current_state = self.model.initial_state
            terminate = False
            run_step_count = 0
            memory_full_at_start = len(self.memory.memory) >= self.memory.memory.maxlen
            avg_q_val = 0
            opt_act_count = 0
            while not terminate:
                action = 0
                if np.random.rand() <= epsilon:


                    action = random.choice(tuple(self.model.valid_actions(current_state)))
                else:
                    opt_act_count += 1
                    q_vals = self.model1.forward(self.tensor.FloatTensor(current_state)).cpu().detach().numpy()
                    val_actions = self.model.valid_actions(current_state)
                    max_idx = None
                    max_q = None
                    for idx, j in enumerate(q_vals):
                        if idx in val_actions:
                            if not max_q:
                                max_q = j
                                max_idx = idx
                            else:
                                if j > max_q:
                                    max_q = j
                                    max_idx = idx
                    avg_q_val += max_q

                    action = max_idx
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                total_reward += reward
                self.memory.append((state, action, next_state, reward, terminal))
                if len(self.memory.memory) >= self.memory.memory.maxlen:
                    self.replay_train()
                current_state = next_state
                run_step_count += 1
                if self.step_per_run is not None and run_step_count >= self.step_per_run:
                    terminate = True
            run_rewards.append(total_reward)

            if memory_full_at_start:
                exec_time = time.time() - start_time
                time_sum.append(exec_time)
                time_avg = mean(time_sum)
                eta = time_avg * (self.runs - run - 1)
                eta_hours = eta / 3600.0
                eta_minutes = int((eta_hours - int(eta_hours)) * 60) + 1
                eta_hours = int(eta_hours)

                print(
                    'Run: {0:8d} || Reward: {1:12.2f} || Epsilon: {2:6.3%} || Avg.Q: {3:6.2f} || Exploitation: {4:3.2%} || ETA: {5:2d} hours {6:2d} minutes'
                        .format(run, total_reward, epsilon, avg_q_val / run_step_count, opt_act_count / run_step_count,
                                eta_hours, eta_minutes))
            else:
                print(
                    'Run: {0:8d} || Reward: {1:12.2f} || Epsilon: {2:6.3%} || Avg.Q: {3:6.2f} || Exploitation: {4:3.2%} || ETA: Estimating...'
                    .format(run, total_reward, epsilon, avg_q_val / run_step_count, opt_act_count / run_step_count))
            self.model.reset_model()


        return run_rewards

    def validate(self, runs, steps_per_run):
        print("Validating policy:")
        run_rewards = []
        for run in range(runs):
            total_reward = 0
            self.model.reset_model()
            current_state = self.model.initial_state
            terminate = False
            run_step_count = 0
            while not terminate:
                action = 0
                q_vals = self.model1.forward(self.tensor.FloatTensor(current_state)).cpu().detach().numpy()
                val_actions = self.model.valid_actions(current_state)
                max_idx = None
                max_q = None
                for idx, j in enumerate(q_vals):
                    if idx in val_actions:
                        if not max_q:
                            max_q = j
                            max_idx = idx
                        else:
                            if j > max_q:
                                max_q = j
                                max_idx = idx
                action = max_idx
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                total_reward += reward
                current_state = next_state
                run_step_count += 1
                if self.step_per_run is not None and run_step_count >= self.step_per_run:
                    terminate = True
            run_rewards.append(total_reward)

            print('Run: {0:8d} || Reward: {1:12.2f}'
                  .format(run, total_reward))
        print("Average reward for ", runs, " runs: ", mean(run_rewards))
        return run_rewards

    def replay_train(self):
        batch = self.memory.sample(self.batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        terminals = []

        for state, action, next_state, reward, terminal in batch:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(terminal)

        states = self.tensor.FloatTensor(states)
        actions = self.tensor.LongTensor(actions)
        next_states = self.tensor.FloatTensor(next_states)
        rewards = self.tensor.FloatTensor(rewards)
        dones = self.tensor.FloatTensor(terminals).view(len(batch), 1)

        curr_q_1 = self.model1.forward(states).gather(1, actions.view(len(batch), 1))
        curr_q_2 = self.model2.forward(states).gather(1, actions.view(len(batch), 1))

        q_next = torch.min(
            torch.max(self.model1.forward(next_states), 1)[0],
            torch.max(self.model2.forward(next_states), 1)[0]
        )


        q_target = rewards.view(len(batch), 1) +(1 - dones)* self.gamma * q_next.view(len(batch), 1)
        self.optizer1.zero_grad()
        loss1 = self.loss1(curr_q_1, q_target.detach())
        loss1.backward()
        self.optizer1.step()

        self.optizer2.zero_grad()
        loss2 = self.loss2(curr_q_2, q_target.detach())
        loss2.backward()
        self.optizer2.step()


    def constant_epsilon(self, run):
        return self.min_epsilon

    def linear_epsilon(self, run):
        return max(self.min_epsilon, 1 - (float(run) / float(self.runs - int(self.runs * self.epsilon_min_percentage))))

    def log_epsilon(self, run):
        return max(self.min_epsilon,
                   ((self.min_epsilon - 1) / ln((1 - self.epsilon_min_percentage) * self.runs)) * ln(run + 1) + 1)

    def gompertz_epsilon(self, run):
        m = int((1 - self.epsilon_min_percentage) * self.runs)
        aux = -2 * ln(((-2 * ln(0.5)) / (m - 1))) / (m - 1)
        ep = 1 - (1 - self.min_epsilon) * exp((-((m - 1) / 2)) * exp(-aux * run))
        return ep if run <= m else self.min_epsilon

    def constant_v2_epsilon(self, run):
        m = int((1 - self.epsilon_min_percentage) * self.runs)
        return self.min_epsilon if run >= m else 0.05

    def cos_epsilon(self, run):
        c = ((((1 - self.min_epsilon) / 2.0) * cos(run * pi / 5)) + (
                self.min_epsilon + ((1 - self.min_epsilon) / 2.0))) * (
                    1 - (run / ((1.0 - self.epsilon_min_percentage) * self.runs)))
        return max(self.min_epsilon, c)

    def logv2(self,run):
        return ((1-self.min_epsilon)*np.exp((-(run**2)-run)*(0.0000015/2)))+self.min_epsilon


class Memory:

    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def append(self, element):
        self.memory.append(element)

    def sample(self, n):
        return random.sample(self.memory, n)