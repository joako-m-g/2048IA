import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

class Agent: 
    def __init__(self, stateSize, actionSize, policyNet, targetNet, replayBuffer, lr=1e-3, gamma=0.99, epsilon=0.05, epsilonDecay=0.999, epsilonMin=0.1):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.policyNet = policyNet
        self.targetNet = targetNet
        self.replayBuffer = replayBuffer

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin

        # Definimos algoritmo de optimizacion y funcion de perdida
        self.optimizer = optim.Adam(self.policyNet.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def selectAction(self, state):
        # Seleccionamos accion con politica epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionSize) # Seleccionamos accion aleatoria
        
        stateTensor = torch.FloatTensor(state).unsqueeze(0).view(-1)
        with torch.no_grad():
            qValues = self.policyNet(torch.log2(stateTensor)) # qvalues: OUTPUT de policyNet
        return qValues.argmax().item() # Accion con mayor valor Q

    def storeTransition(self, state, action, reward, nextState, done):
        # Almacenamos la transicion en el Replay Buffer
        self.replayBuffer.push(state, action, reward, nextState, done)
    
    def train(self, batchSize):
        # Entrenamos la red usando muestras aleatorias del Replay Buffer
        if len(self.replayBuffer) < batchSize:
            return # No hay suficientes transiciones para entrenar

        states, actions, rewards, nextStates, dones = self.replayBuffer.sample(batchSize)  # Tomo una muestra de transiciones de tamaño 'batchSize'

        # Q valores actuales
        qValues = self.policyNet(torch.log2(states)).gather(1, actions.unsqueeze(1))
        # Q valores objetivo - Se usa no_grad porque esta red se actualiza después de X pasos
        with torch.no_grad():
            nextQValues = self.targetNet(torch.log2(nextStates)).max(1, keepdim=True)[0]
            targetQValues = rewards.unsqueeze(1) + self.gamma * nextQValues * (1 - dones.unsqueeze(1))

        loss = self.criterion(qValues, targetQValues)  # Calculamos la pérdida
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()  # Realizamos la retropropagación
        self.optimizer.step()  # Actualizamos los pesos

    def updateEpsilon(self): 
        # Reducimos el epsiolon
        self.epsilon = max(self.epsilon * self.epsilonDecay, self.epsilonMin)
        return self.epsilon

    def updatetargetNet(self): 
        # Actualizamos pesos de red objetivo
        self.targetNet.load_state_dict(self.policyNet.state_dict())


    