import sys
sys.path.append("C:/Users/pc/Desktop/2048IA") # Agregamos directorio raiz del proyecto
import numpy as np
import csv
import torch
from game.dosmil import Game
from model.agent import Agent
from model.networks import QNetwork
from model.replayBuffer import ReplayBuffer
import os
import pandas as pd

class DQNTrainer:
    """
    Clase para entrenar un agente usando Deep Q-Learning (DQN).
    """

    def __init__(self, stateSize, actionSize, gamma=0.99, epsilon=0.05, epsilonDecay=0.9999,
                 learningRate=1e-3, batchSize=1024, replayBufferSize=100000, updateTargetFreq=3000):
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
        self.redes = QNetwork(stateSize, actionSize)
        self.policyNet, self.targetNet = self.redes.createNetwork(stateSize, actionSize)

        # Cargamos los pesos de la ultima sesion de entrenamiento
        if len(os.listdir('training')) >= 3: # Verificamos si tenemos los estdos anteriores guardados
            print('Buscando estado guardado del modelo...')
            self.redes.loadModels(self.policyNet, self.targetNet, 'training/modeloEntrenadoPolicy.pth', 'training/modeloEntrenadoTarget.pth')
            df = pd.read_csv('training/results.csv')
            self.epsilon = df.iloc[-1]['epsilon']

        # Inicializamos Replay Buffer
        self.replayBuffer = ReplayBuffer(replayBufferSize)

        # Inicializamos el entorno y el agente
        self.env = Game()
        self.agent = Agent(stateSize, actionSize, self.policyNet, self.targetNet, self.replayBuffer,
                           lr=learningRate, gamma=gamma, epsilon=self.epsilon, epsilonDecay=epsilonDecay)

        # Lista de acciones posibles y su representación numérica
        self.actions = ['izquierda', 'derecha', 'arriba', 'abajo']
        self.numericAction = {'izquierda': 0, 'derecha': 1, 'arriba': 2, 'abajo': 3}

    def train(self, numEpisodios=10000):
        """
        Entrena al agente en un número dado de episodios.
        """
    
        # Definimos numero de episodios
        for episodio in range(numEpisodios):
            metrics = {} # Creo diccionario para las metricas
            metrics['episodio'] = episodio
            tablero = None
            reward = 0
            movs = 0
            cumReward = 0
            tablero = self.env.crear_tablero(4)
            movsCount = {'izquierda': 0, 'derecha': 0, 'arriba': 0, 'abajo': 0}
            
            # Recolección de datos (experiencia)
            while not self.env.esta_atascado(tablero):
                movs += 1
                tablero = self.env.llenar_pos_vacias(tablero, 1)
                state = tablero.copy()
                action = self.actions[self.agent.selectAction(self.to_one_hot(self.tablero_a_log2(state).flatten()).flatten())] # Selección de acción
                movsCount[action] += 1 # Contamos la accion
                tablero = self.env.mover(tablero, action)
                nextState = tablero.copy()
                reward = self.env.reward(tablero, state)
                cumReward += reward # Calculamos recompensa acumulada
                done = not self.env.esta_atascado(tablero)  # Verifica si el episodio terminó
                self.agent.storeTransition(self.to_one_hot(self.tablero_a_log2(state).flatten()).flatten(), self.numericAction[action], reward, self.to_one_hot(self.tablero_a_log2(nextState).flatten()).flatten(), done)  # Guardamos la experiencia (transicion)
                
            print(tablero)
            # Calculamos metricas de la partida
            metrics['puntajeTotal'] = np.sum(tablero)
            metrics['puntajeMean'] = np.mean(tablero)
            metrics['bestFicha'] = np.max(tablero)
            metrics['movimientos'] = movs
            metrics['cantMovimientos'] = movsCount
            metrics['epsilon'] = self.epsilon
            metrics['cumReward'] = cumReward
            
            self.storeMetrics(metrics) #Guardamos metricas en csv
            self.epsilon = self.agent.updateEpsilon()  # Actualizamos epsilon

            for _ in range(10):
                self.agent.train(self.batchSize) # Entrenamiento de las redes

            # Actualización de la red objetivo a intervalos definidos
            if episodio % 50 == 0:
                print('Guardando modelo...')
                self.saveModel([self.policyNet, self.targetNet])
                #print("Actualizamos red objetivo")
            if episodio % self.updateTargetFreq == 0:
                print('Actualizamos red targert...')
                self.agent.updatetargetNet()
    
    def tablero_a_log2(self, tablero):
        # Crear una copia del tablero para no modificar el original
        tablero_log2 = np.zeros_like(tablero, dtype=int)
        # Aplicar log2 solo donde el tablero tiene valores mayores a 0
        tablero_log2[tablero > 0] = np.log2(tablero[tablero > 0]).astype(int)
        return tablero_log2
    
    def to_one_hot(self, vector):
        vector = np.array(vector)  # Asegurarse de que sea un array de NumPy
        max_classes = 16 + 1  # Máximo valor posible es log2(65536) = 16, entonces 17 clases
        one_hot_matrix = np.eye(max_classes)[vector]  # Generar la matriz one-hot
        return one_hot_matrix
    
    def storeMetrics(self, metrics):
        # Verifica si el archivo ya existe
        archivo_existe = False
        try:
            with open('training/results.csv', "r"):
                archivo_existe = True
        except FileNotFoundError:
            archivo_existe = False
        
        # Guardamos metricas del episodio en un CSV
        with open('training/results.csv', mode='a') as file:
            csv_writer = csv.DictWriter(file, fieldnames=metrics.keys())
            # Si el archivo no existe, escribe el encabezado
            if not archivo_existe:
                csv_writer.writeheader()
            csv_writer.writerow(metrics)
        
    def saveModel(self, models):
        # Guardamos pesos de la red neuronal
        torch.save(models[0].state_dict(), 'training/modeloEntrenadoPolicy.pth')
        torch.save(models[1].state_dict(), 'training/modeloEntrenadoTarget.pth')


# Uso de la clase DQNTrainer
if __name__ == "__main__":
    # Definir el tamaño del estado y las acciones
    stateSize = 272  # Características del estado (tamaño del tablero)
    actionSize = 4  # Número de acciones posibles

    # Inicializar y entrenar el agente
    trainer = DQNTrainer(stateSize, actionSize)
    trainer.train(numEpisodios=70000)
