import numpy as np
import torch 
import pygame
import stable_baselines3
from mydqn_components import ReplayBuffer
from mydqn_components import Q_network
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn

class Baseline():
    def __init__(self, difficult=1, right_play=1):
        self.difficult = difficult   # Сделаем для моделей уровни сложности 
        self.right_play = right_play
        if (difficult < 0) or (difficult>2):
            raise ValueError(f"difficult must be in (0, 2) (0.8 - easy, 1 - medium, 1.2 - hard)")
    def act(self, state, envs_params):
        ball_y, ball_x, vy, vx, paddle_agent_y, paddle_opponent_y = state
        field_width, field_height, paddle_width, paddle_height, ball_size, ball_speed = envs_params
        # Режимы: right_play = 1, а если слева - то "-1"         
        if self.right_play == 1:
            if vx > 0:   # мяч на эвристику которая справа (кстати опастно если эвристика справа, то все сломается)
                if ball_y > paddle_opponent_y + paddle_height/2:
                    return 0
                else:
                    return 1
            else:
                if paddle_opponent_y + paddle_height/2 < field_height/2:
                    return 0
                else:
                    return 1
        else:
            if vx < 0:   # мяч на эвристику которая справа (кстати опастно если эвристика справа, то все сломается)
                if ball_y > paddle_agent_y + paddle_height/2:
                    return 0
                else:
                    return 1
            else:
                if paddle_agent_y + paddle_height/2 < field_height/2:
                    return 0
                else:
                    return 1

class DQN_StBaselines3():
    def __init__(self, difficult=1, model=None):
        self.difficult = difficult
        self.model= model
        if (difficult < 0) or (difficult>2):
            raise ValueError(f"difficult must be in (0, 2) (0.8 - easy, 1 - medium, 1.2 - hard)")
    def act(self, state, envs_params):
        # Проверка: если модель есть - используем её
        if self.model is not None:
            action, _ = self.model.predict(state, deterministic=True)
            return int(action)
        else:
            raise KeyError("Model not fit yet.")


class HumanPlayer():
    def __init__(self, difficult=1.0):
        """
        difficult: множитель скорости ракетки (0.8 - медленнее, 1.0 - норма, 1.2 - быстрее)
        """
        self.difficult = difficult
        if (difficult < 0) or (difficult>2):
            raise ValueError(f"difficult must be in (0, 2) (0.8 - easy, 1 - medium, 1.2 - hard)")

    def act(self, state, envs_params):
        """
        Читает нажатия клавиш и возвращает действие.
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 1  # вверх
        elif keys[pygame.K_DOWN]:
            return 0  # вниз
        else:
            return 2  # стой

class MyDQN():
     def __init__(
        self,
        q_network,
        target_network=None,           
        learning_rate=1e-3,           
        buffer_capacity=50000,       
        batch_size=64,                 
        gamma=0.99,                   
        target_update_freq=1000,      
        learning_starts=5000,
        device='cpu',                  # устройство ('cpu' или 'cuda' (gpu))
        beta = lambda x : 1/np.log(x + 2),
        n_samples = 10,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = q_network.to(self.device)
        self.target_network = target_network.to(self.device) if target_network is not None else self._copy_network(q_network).to(self.device)
        self.lr = learning_rate
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts  # сколько нужно в буфере чтобы начинать учиться
        self.beta = beta
        self.n_samples = n_samples
        self.step_counter = 0
        self.episode_counter = {'agent' : 0,
                                'baseline' : 0}
        self.buffer = ReplayBuffer(batch_size=self.batch_size, capacity=self.buffer_capacity)
        self.optimizer = Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_history = []
        
     def act(self, state, envs_params):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.q_network.train()
        with torch.no_grad():
            preds = [self.q_network(state_t) for _ in range(self.n_samples)]
        q_samples = torch.cat(preds, dim=0) 
        mean_q = q_samples.mean(dim=0)
        std_q = q_samples.std(dim=0)
        return torch.argmax(mean_q + self.beta(self.step_counter) * std_q).item()

     def update(self):
         
        if len(self.buffer) >= self.learning_starts:
            batch = self.buffer.sample()
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

            q_values = self.q_network(states)
            q_s_a = q_values[range(self.batch_size), actions]
            
            with torch.no_grad():
                self.target_network.eval()
                max_next_q = self.target_network(next_states).max(dim=1)[0]
                targets = rewards + self.gamma * max_next_q * (1-dones)
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(q_s_a, targets)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()          
        else:
            return 0.0
     def _copy_network(self, network):
        copy = Q_network(
            input_size=network.fc1.in_features,
            hidden_size=network.fc2.in_features,
            action_size=network.fc3.out_features,
            func_activation=network.activation,
            dropout_proba=network.dropout_proba
        )
        copy.load_state_dict(network.state_dict())
        return copy.to(self.device)
     def learn(self, env, total_timesteps = 10000, alpha = 0.15):
        current_state, _ = env.reset()
        rew_on_lr = []
        rew_on_episode = 0.0
        for i in range(total_timesteps):
            action = self.act(current_state, env.envs_params)
            new_state, reward, done, _, _  = env.step(action)
            rew_on_episode += reward
            self.buffer.push((current_state, action, reward, new_state, done))
            cur_loss = self.update()
            current_state = new_state
            if (i % 1000 == 0) and i > (self.learning_starts):
                self.loss_history.append(cur_loss)
            if self.step_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            self.step_counter += 1
            if done:
                rew_on_lr.append(rew_on_episode if not rew_on_lr else (1-alpha)*rew_on_lr[-1] + alpha*rew_on_episode)
                rew_on_episode = 0.0
                current_state, _ = env.reset()
        return rew_on_lr    # по нему можно построить график как в лекции от шада
                
     def pretrain_on_dataset(self, data, n_epoch = 10):
        self.q_network.train()
        X = torch.tensor(data[['ball_y', 'ball_x', 'vy', 'vx', 'paddle_agent_y', 'paddle_opponent_y']].values, dtype=torch.float32)
        y = torch.tensor(data['action'].values, dtype = torch.long)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle= True)
        criterion = nn.CrossEntropyLoss()
        
        for i in range(n_epoch):
            ep_loss = 0.0
            for X_batch, y_batch in dataloader:
                
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.q_network(X_batch)

                self.optimizer.zero_grad()
                loss = criterion(preds, y_batch)
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            if i % 10 == 0:
                print(f"{ep_loss/(len(dataloader))} - loss on {i+1}-th epoch")
                
     def save(self, filename):
        path = "saved_models/" + filename
        torch.save(self.q_network.state_dict(), path)
     def load(self, filename):
        path = "saved_models/" + filename
        self.q_network.load_state_dict(torch.load(path, map_location = self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.train()