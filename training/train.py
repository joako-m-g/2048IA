import sys
sys.path.append("C:/Users/pc/Desktop/2048IA") # Agregamos directorio raiz del proyecto
import numpy as np
import csv
import torch
from game.dosmil import Game
from model.agent import Agent
from model.networks import QNetwork
from model.replayBuffer import ReplayBuffer
import math




class DQNTrainer:
    """
    Clase para entrenar un agente usando Deep Q-Learning (DQN).
    """

    def __init__(self, stateSize, actionSize, gamma=0.97, epsilon=0.1, epsilonDecay=0.9999,
                 learningRate=1e-3, batchSize=64, replayBufferSize=500000, updateTargetFreq=500):
        """
        Inicializa los hiperparámetros, las redes, el agente y el entorno.
        """
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.replayBufferSize = replayBufferSize
        self.updateTargetFreq = updateTargetFreq

        # Inicialización de las redes y el agente
        redes = QNetwork(stateSize, actionSize)
        self.policyNet, self.targetNet = redes.createNetwork(stateSize, actionSize)

        # Inicializamos Replay Buffer
        self.replayBuffer = ReplayBuffer(replayBufferSize)

        # Inicializamos el entorno y el agente
        self.env = Game()
        self.agent = Agent(stateSize, actionSize, self.policyNet, self.targetNet, self.replayBuffer,
                           lr=learningRate, gamma=gamma, epsilon=epsilon, epsilonDecay=epsilonDecay)

        # Lista de acciones posibles y su representación numérica
        self.actions = ['izquierda', 'derecha', 'arriba', 'abajo']
        self.numericAction = {'izquierda': 0, 'derecha': 1, 'arriba': 2, 'abajo': 3}

    def play(self):
        """
        Realiza un episodio de prueba para evaluar la política del agente.
        """
        reward = 0
        tablero = self.env.crear_tablero(4)  # Creamos el tablero

        while not self.env.esta_atascado(tablero):
            tablero = self.env.llenar_pos_vacias(tablero, 1)
            ogTablero = tablero.copy()
            action = self.actions[self.agent.maximiza(ogTablero)]  # Selecciona la acción usando la política
            tablero = self.env.mover(tablero, action)
            reward = self.env.reward(tablero, ogTablero, action)
            #print(tablero)
    
            # Si el tablero no cambió, terminamos el juego
            if np.array_equal(ogTablero, tablero):
                #print("La acción seleccionada no cambió el estado.")
                print(tablero)
                return np.sum(tablero)
        print(tablero)

        return reward

    def train(self, numEpisodios=60000):
        """
        Entrena al agente en un número dado de episodios.
        """
        # Definimos 50000 episodios de recolección de datos
        for episodio in range(numEpisodios):
            tablero = None
            reward = 0

            """
            number = np.random.randint(1, 5)
            if number == 1:
                tablero = self.env.tableroAlineadoHorizontal(4)
            elif number == 2:
                tablero = self.env.tableroAleatorio(4)
            elif number == 3:
                tablero = self.env.crear_tablero(4)  # Tableros vacíos
            elif number == 4:
                tablero = self.env.tableroAlineadoVertical(4)
            """
            tablero = self.env.crear_tablero(4)
            
            # Recolección de datos (experiencia)
            while not self.env.esta_atascado(tablero):
                tablero = self.env.llenar_pos_vacias(tablero, 1)
                state = tablero.copy()
                action = self.actions[self.agent.selectAction(state)]  # Selección de acción
                tablero = self.env.mover(tablero, action)
                nextState = tablero.copy()
                reward = self.env.reward(tablero, state, action)
                done = not self.env.esta_atascado(tablero)  # Verifica si el episodio terminó
                self.agent.storeTransition(state, self.numericAction[action], reward, nextState, done)  # Guardamos la experiencia

            
            # Entrenamiento de las redes
            self.agent.train(self.batchSize)

            self.agent.updateEpsilon()  # Actualizamos epsilon

            # Actualización de la red objetivo a intervalos definidos
            if episodio % self.updateTargetFreq == 0:
                #print("Actualizamos red objetivo")
                self.agent.updatetargetNet()

            # Jugamos y evaluamos la mejora del modelo
            recompensa = self.play()
            #print("Recompensa del episodio:", recompensa)

            # Guardamos la recompensa del episodio en un archivo CSV
            with open('results.csv', mode='a') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([recompensa, episodio])
        
        self.saveModel(self.policyNet)

    def saveModel(self, model):
        # Guardamos pesos de la red neuronal
        torch.save(model.state_dict(), 'modeloEntrenado.pth')


# Uso de la clase DQNTrainer
if __name__ == "__main__":
    # Definir el tamaño del estado y las acciones
    stateSize = 16  # Características del estado (tamaño del tablero)
    actionSize = 4  # Número de acciones posibles

    # Inicializar y entrenar el agente
    trainer = DQNTrainer(stateSize, actionSize)
    trainer.train(numEpisodios=50000)
