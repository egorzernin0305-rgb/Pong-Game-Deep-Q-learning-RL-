import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import models
import pandas as pd


class PongEnv(gym.Env):
    def __init__(self, render_mode = None, opponent=models.Baseline(difficult=1), agent_side='left', n_rounds=10):
        super().__init__()

        self.agent_side = agent_side
        self.opponent = opponent
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
        self.screen = None
        self.clock = None
    
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

        self.field_width = 800
        self.field_height = 600    
        self.paddle_width = 10
        self.paddle_height = 80
        self.ball_size = 10
        self.ball_speed = 4
        self.paddle_speed = 6
        self.paddle_agent_x = self.paddle_width # Расстояние в пикселях от левого края до ракетки агента (до левой части ракетки)
        self.paddle_opponent_x = self.field_width - self.paddle_width * 2
        self.max_speed = 2*self.ball_speed    # Причем 2*.. достигается если отбили краем ракетки
        self.state = None
        self.envs_params = [
        self.field_width / self.field_width,
        self.field_height / self.field_height,
        self.paddle_width / self.field_width,
        self.paddle_height / self.field_height,
        self.ball_size / self.field_height,
        self.ball_speed / self.ball_speed
        ]
        self.scoreboard = {'agent' : 0, 'opponent' : 0}
        self.n_rounds = n_rounds
        # Нормализованные параметры среды
        
    def reset(self, seed=None, options=None, reset_scoreboard= False):
        super().reset(seed=seed)
        if reset_scoreboard:
            self.scoreboard = {'agent' : 0, 'opponent' : 0}
        if seed is not None:
            np.random.seed(seed)

        self.ball_y = np.clip(np.random.normal(self.field_height // 2, 20), 20, self.field_height - 20)   
        self.ball_x = self.field_width // 2
        self.vy = self.ball_speed * np.random.choice([-1, 1]) * np.random.uniform(0.7, 1.2)
        self.vx = self.ball_speed * np.random.choice([-1, 1])
        self.paddle_agent_y = self.field_height // 2 - self.paddle_height // 2
        self.paddle_opponent_y = self.field_height // 2 - self.paddle_height // 2
        obs = np.array([self.ball_y/self.field_height, self.ball_x/self.field_width, self.vy / self.max_speed, self.vx / self.max_speed,  self.paddle_agent_y / self.field_height,  self.paddle_opponent_y / self.field_height], dtype=np.float32)
        self.state = obs
        return obs, self.scoreboard
    def step(self, action):
        ## Действие Агента
        if action == 0:
            self.paddle_agent_y += self.paddle_speed       #вниз
        elif action == 1:
            self.paddle_agent_y -= self.paddle_speed       #вверх
            
        self.paddle_agent_y = np.clip(
        self.paddle_agent_y,
        self.paddle_width,
        self.field_height - self.paddle_width - self.paddle_height
        )

        opp_act = self.opponent.act(self.state, self.envs_params)
        ## Действие оппонента  
        if opp_act == 0:
            self.paddle_opponent_y += self.opponent.difficult*self.paddle_speed       #вниз
        elif opp_act == 1:           #вверх
            self.paddle_opponent_y -= self.opponent.difficult*self.paddle_speed
                
        # Ограничение
        self.paddle_opponent_y = np.clip(
        self.paddle_opponent_y,
        self.paddle_width,
        self.field_height - self.paddle_width - self.paddle_height
        )

        
        ## На самом деле если мяч окажется по середине ракетки, то она дернется вверх, но это событие маловероятно и не сильно влияет на исход игры.
        
        ## PS. Пока что эвристика просто следует за мячом, потом можно сделать какую-нибудь
        ## модель и обучить ее на собранном датасете (или найти на Kaggle если найдется подходящий)
        # Ограничение, чтобы ракетка не выходила за поле

        self.ball_x += self.vx
        self.ball_y += self.vy
                
        reward = 0.0
        terminated = False
        truncated = False
        
        cond_agent_hitb = ((self.vx < 0) and (self.ball_x - self.ball_size/2 <= self.paddle_agent_x + self.paddle_width) and (self.paddle_agent_y <= self.ball_y and self.ball_y <= self.paddle_agent_y + self.paddle_height))
        cond_opponent_hitb = ((self.vx > 0) and (self.ball_x + self.ball_size/2 >= self.paddle_opponent_x) and (self.paddle_opponent_y <= self.ball_y and self.ball_y <= self.paddle_opponent_y + self.paddle_height)) ## - self.paddle_width
        cond_ball_hit_topwall = ((self.vy < 0) and (self.ball_y <= self.ball_size/2 + self.paddle_width))
        cond_ball_hit_downwall = ((self.vy > 0) and (self.ball_y >= self.field_height - self.ball_size/2 - self.paddle_width))

        if (cond_ball_hit_topwall):
            self.vy = -self.vy
        if (cond_ball_hit_downwall):
            self.vy = -self.vy

        if (cond_agent_hitb):
            alpha = ((self.ball_y - self.paddle_agent_y) - self.paddle_height/2)/(self.paddle_height/2)
            if action == 2:  # ракетка стоит
                self.vx = -self.vx
            else:
                self.vx = np.sign(self.vx)*self.ball_speed      #сбрасываем ускорение с которым отбил оппонент
                self.vy = np.sign(self.vy)*self.ball_speed
                self.vy = self.vy * (1 + alpha**2)
                self.vx = -self.vx* (1 + (1 - alpha**2))   
            #reward += 0.3*np.exp(abs(alpha))  ## Агент больше учится отбивать краем ракетки
            reward += 0.15 + 0.1*abs(alpha)

        if (cond_opponent_hitb):
            self.vx = np.sign(self.vx)*self.ball_speed      #сбрасываем ускорение с которым отбил агент
            self.vy = np.sign(self.vy)*self.ball_speed
            alpha = ((self.ball_y - self.paddle_opponent_y) - self.paddle_height/2)/(self.paddle_height/2)
            self.vy = self.vy * (1 + alpha**2)
            self.vx = -self.vx* (1 + (1 - alpha**2))   


        if (self.ball_x < 0):
            reward += -1   # Нам забили
            terminated = True
            self.scoreboard['opponent'] += 1
            if self.scoreboard['opponent'] >= self.n_rounds:
                truncated = True
        if (self.ball_x > self.field_width):
            reward += 1    # Мы забили
            terminated = True
            self.scoreboard['agent'] += 1
            if self.scoreboard['agent'] >= self.n_rounds:
                truncated = True
        self.state= np.array([self.ball_y/self.field_height, self.ball_x/self.field_width, self.vy / self.max_speed, self.vx / self.max_speed,  self.paddle_agent_y / self.field_height,  self.paddle_opponent_y / self.field_height], dtype=np.float32)

        return self.state, reward, terminated, truncated, self.scoreboard

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.field_width, self.field_height)
                )
                pygame.display.set_caption("Pong RL")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Arial", 36)  # шрифт для счёта
    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
    
            self.screen.fill((0, 0, 0))
    
            # Белые границы (каёмка)
            border_width = self.paddle_width  # 10px
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (0, 0, self.field_width, border_width)  # верхняя
            )
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (0, self.field_height - border_width, self.field_width, border_width)  # нижняя
            )
    
            # Мяч
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),
                (int(self.ball_x), int(self.ball_y)),
                self.ball_size // 2
            )
    
            # Левая ракетка (агент)
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (self.paddle_agent_x, self.paddle_agent_y,
                 self.paddle_width, self.paddle_height)
            )
    
            # Правая ракетка (оппонент)
            pygame.draw.rect(
                self.screen,
                (0, 0, 255),
                (self.paddle_opponent_x, self.paddle_opponent_y,
                 self.paddle_width, self.paddle_height)
            )
    
            # Счёт
            score_text = f"{self.scoreboard['agent']} : {self.scoreboard['opponent']}"
            text_surface = self.font.render(score_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.field_width // 2, 30))
            self.screen.blit(text_surface, text_rect)
    
            pygame.display.flip()
            self.clock.tick(60)
            
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None

    def demo(self, left_player=models.Baseline(difficult=1, right_play=-1)):
        # Настройка левого игрока
        if isinstance(left_player, models.Baseline):
            left_player.right_play = -1
        
        # self.opponent уже задан в __init__
        _, _ = self.reset(reset_scoreboard=True)
        truncated = False
        reward_for_round = 0.0
        rewards = []
        steps_count_for_round = 0
        steps_count = []
        while not truncated:
            action = left_player.act(self.state, self.envs_params)
            obs, reward, terminated, truncated, info = self.step(action)
            reward_for_round += reward
            steps_count_for_round += 1
            self.render()
            
            if terminated or truncated:
                rewards.append(reward_for_round)
                steps_count.append(steps_count_for_round)
                reward_for_round = 0.0
                steps_count_for_round = 0
                _, _ = self.reset()
        rewards = np.array(rewards)
        steps_count = np.array(steps_count)
        stats = {
            'avg_round_steps' : steps_count.mean(),
            'avg_round_time' : (steps_count.mean())/60,
            'avg_round_reward' : rewards.mean(),
            'variance_reward' : rewards.var(),
            'score' : self.scoreboard,
            'agent_win_rate' : self.scoreboard['agent']/(self.scoreboard['agent'] + self.scoreboard['opponent']),
            'rounds_amount' : self.scoreboard['agent'] + self.scoreboard['opponent']
        }
        self.close()
        return stats
        
## action = 2 - stay
## action = 0  1 двигаемся (вверх = 1) (вниз = 0)
