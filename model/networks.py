import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Esta clase define la red neuronal que usaremos para la politica y la red objetivo.
    Ambas redes son iguales en estructura.
    """

    def __init__(self, stateSize, actionSize):
        super(QNetwork, self).__init__()

        # Capa de entrada que toma el tamaño del estado (cant de estados)
        self.fc1 = nn.Linear(stateSize, 1024) # Primera capa fullyConnected
        self.fc2 = nn.Linear(1024, 1024) # Segunda capa fullyConnected
        self.fc3 = nn.Linear(1024, 1024) # Tercera capa fullyConnected
        self.fc4 = nn.Linear(1024, 1024) # Cuarta capa fullyConnected
        self.fc5 = nn.Linear(1024, actionSize) # Capa de salida (tamaño igual al numero de acciones)

    def forward(self, state): 
        """
        Define el paso hacia adelante para la red neuronal. 
        Recibe in estado y devuelve los valores Q para cada accion.
        """
        x = F.relu(self.fc1(state)) # Activacion ReLu en la primera capa
        x = F.relu(self.fc2(x)) # Activacion ReLu en la segunda capa
        x = F.relu(self.fc3(x)) # Activacion ReLu en la segunda capa
        x = F.relu(self.fc4(x)) # Activacion ReLu en la segunda capa
        qValues = self.fc5(x) # Los valores Q para cada accion

        return qValues
    
    def createNetwork(self, stateSize, actionSize):
        """
        Crea e incializa las redes policy y target.
        Ambas redes son instanciasd de la QNetwork, pero targetNet no se actualizara
        de manera contunua durante el entrenamiento (lleva un retraso en sus pesos) 
        """
        # Crear la red de politica
        policyNet = QNetwork(stateSize, actionSize)

        # Crear la red objetivo y copiar los parametros de la red de politica
        targetNet = QNetwork(stateSize, actionSize)
        targetNet.load_state_dict(policyNet.state_dict()) # Cargamos en la targetNet diccionario con los parametros de la policyNet

        return policyNet, targetNet

    def loadModels(self, policyNet, targeNet, policyPath, targetPath):
        policyNet.load_state_dict(torch.load(policyPath)) # Cargamos los pesos a la primera red
        targeNet.load_state_dict(torch.load(targetPath)) # Cargamos los pesos a la segunda red
        