import numpy as np
import torch 
import pygame
import stable_baselines3


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