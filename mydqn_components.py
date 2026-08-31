import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pandas as pd 

class ReplayBuffer:
  def __init__(self, batch_size = 64, random_state = 42, capacity = 50000):
    self.batch_size = batch_size
    self.random_state = random_state
    self.capacity = capacity
    self.buffer = deque(maxlen = self.capacity)
    self.rng = random.Random(random_state)
  def sample(self, size= None):
    if not size:
        size = self.batch_size
    return self.rng.sample(self.buffer, size)
  def push(self, element):
    self.buffer.append(element)
    pass
  def __len__(self):
    return len(self.buffer)
class Q_network(nn.Module):
  def __init__(self, input_size=6, hidden_size=128, action_size=3, func_activation = torch.relu, dropout_proba = 0.4):
    super().__init__()
    self.dropout_proba = dropout_proba
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, action_size)
    self.dropout = nn.Dropout(dropout_proba)
    self.activation = func_activation
  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.dropout(x)
    x = self.activation(self.fc2(x))
    x = self.dropout(x)
    return self.fc3(x)
