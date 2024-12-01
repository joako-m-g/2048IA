import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # Deque es mas eficiente que las listas para pop(0)

    def push(self, state, action, reward, nextState, done):
        # Añadimos nueva transicion al buffer
        self.buffer.append((state, action, reward, nextState, done))
    
    def sample(self, batchSize): 
        # Muestra aleatoria del buffer de tamaño 'batchSize'
        batch = random.sample(self.buffer, batchSize)
        
        states, actions, rewards, nextStates, dones = zip(*batch)

        # Convertimos a tensores en PyTorch
        states = torch.tensor(np.array([state.flatten() for state in states]), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        nextStates = torch.tensor(np.array([state.flatten() for state in nextStates]), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, rewards, nextStates, dones

    def __len__(self): 
        # Devuelve el numero de transiciones en el buffer
        return len(self.buffer)


