import torch
from game.gui import GameGUI
from model.networks import QNetwork

class Test:
    def __init__(self, stateSize, actionSize)
        self.stateSize = stateSize
        self.actionSize = actionSize

    def loadModel(self): 
        model = QNetwork(self.stateSize, self.actionSize)
        model.load_state_dict(torch.load('modeloEntrenado.pth'))
        model.eval()

        return model

    def play(tablero):
        pass

    # IMPLEMENTAR CORRIDA DEL JUEGO CON CONSULTAS AL MODELO #   