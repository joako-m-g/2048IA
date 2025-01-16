import random
import numpy as np
import math

class Game:
    def  crear_tablero(self, n):
        tablero = np.zeros((n, n))
        return tablero

    def listar_pos_vacias(self, tablero):
        lista=[]
        for i in range(tablero.shape[0]):
            for j in range(tablero.shape[1]):
                if tablero[(i,j)]==0: lista.append((i,j))
        return lista

    def llenar_pos_vacias(self, tablero, cantidad):
        lista=self.listar_pos_vacias(tablero)
        random.shuffle(lista)
        i=0
        while i<len(lista) and i<cantidad:
            tablero[lista[i]]=2
            i+=1
        return tablero
            
    def mover_fichas_un_lugar(self, fila):
        for i in range(1,len(fila)):
            if fila[i] != 0 and fila[i-1]==0: 
                fila[i-1] = fila[i]
                fila[i] = 0
        return fila

    def colapsar_fichas(self, fila):
        fila_movida = self.mover_fichas(fila)
        fila_sumada = self.sumar_adyacentes_iguales(fila_movida)
        fila_sumada= self.mover_fichas(fila)
        return fila_sumada

    def colapsar(self, tablero):
        tablero_c=tablero.copy()
        for i in range(tablero.shape[0]):
            tablero[i,:]=self.colapsar_fichas(tablero[i,:])
        if not(tablero==tablero_c).all():
            tablero=self.llenar_pos_vacias(tablero,1)
        return tablero

    def mover_fichas(self, fila):
        stop=False
        while not stop:
            fila_i=fila.copy()
            fila=self.mover_fichas_un_lugar(fila)
            fila_s=np.array(fila)
            fila_a=np.array(fila_i)
            if (fila_s==fila_a).all():
                stop=True       
        return fila
                
    def sumar_adyacentes_iguales(self, fila):
        for i in range(len(fila)-1):
            if fila[i] == fila[i+1]:
                fila[i] = fila[i] + fila[i+1]
                fila[i+1] = 0
        return fila


    def mover(self, tablero, direccion):
        rotaciones={"izquierda": 0,"abajo":3 ,"derecha":2, "arriba":1}
        for i in range(rotaciones[direccion]):
            tablero=np.rot90(tablero,1)
        tablero=self.colapsar(tablero)
        for i in range(rotaciones[direccion]):
            tablero=np.rot90(tablero,-1)
        return tablero


    def esta_atascado(self, tablero):
        devolucion = True
        pos_vacias = self.listar_pos_vacias(tablero)
        if len(pos_vacias) == 0:
            i,j=(0,0)
            while i<tablero.shape[0] and devolucion:
                while j<tablero.shape[1] and devolucion:
                    # Interior del tablero
                    if (i+j)%2 == 0 and i != 0 and j != 0 and i != tablero.shape[0]-1 and j != tablero.shape[1]-1:
                        if tablero[i,j] == tablero[i-1,j] or tablero[i,j] == tablero[i+1,j] or tablero[i,j]==tablero[i,j-1] or tablero[i,j] ==tablero[i,j+1]:
                            devolucion = False
                    j+=1
                i+=1

            # rotamos el tablero para verificar los bordes
            for x in range(4):
                tablero=np.rot90(tablero, -1)
                if tablero[0,0] == tablero[1,0] or tablero[0,0] == tablero[0, 1]:
                    devolucion = False
                for j in range(tablero.shape[1]-1): #columnas
                    if tablero[0,j] == tablero[1,j] or tablero[0,j] == tablero[0, 1+j]:
                        devolucion = False
        else: devolucion=False
        return devolucion
    
    def tableroAleatorio(self, n):
        tablero = self.crear_tablero(n)
        for i in range(n):
            for j in range(n):
                tablero[i, j] = 2**random.randint(1, 8)
        return tablero
        
    def tableroAlineadoHorizontal(self, n):
        tablero = self.crear_tablero(n)
        for i in range(n):
            Power = random.randint(1, 8)
            for j in range(n):
                tablero[i, j] = 2** Power
        return tablero
    
    
    def tableroAlineadoVertical(self, n):
        # Crear una matriz vacía de tamaño nxn
        tablero = np.zeros((n, n), dtype=int)
        
        # Generar valores para las columnas en espejo
        columnas = []
        for j in range((n + 1) // 2):  # Solo hasta la mitad (incluso si n es impar)
            power = random.randint(1, 8)  # Potencias de 1 a 8
            columnas.append(2 ** power)
        
        # Rellenar columnas en espejo
        for j in range(n):
            if j < n // 2:
                tablero[:, j] = columnas[j]  # Primera mitad
            else:
                tablero[:, j] = columnas[n - j - 1]  # Segunda mitad (espejo)
        
        return tablero
                
    def tablero_ordenado(self, tablero):
        # Verificar filas (ascendente o descendente)
        for fila in tablero:
            if not (np.all(fila[:-1] <= fila[1:]) or np.all(fila[:-1] >= fila[1:])):
                return False

        # Verificar columnas (ascendente o descendente)
        for col in tablero.T:
            if not (np.all(col[:-1] <= col[1:]) or np.all(col[:-1] >= col[1:])):
                return False

        return True

    
    # Calculo de recompensa
    def reward(self, tablero, lastTablero):
        reward = 0

        # Premio por colapsar fichas (incremento en valores del tablero)
        for i in range(tablero.shape[0]):
            for j in range(tablero.shape[1]):
                if tablero[i, j] > lastTablero[i, j]:
                    reward += math.log2(tablero[i, j]) if tablero[i, j] != 0 else 0

        # Penalización por estancamiento
        if np.array_equal(tablero, lastTablero):
            # Aumenta la penalización por estancamiento consecutivo
            if not self.esta_atascado(tablero):
                numDifCero = tablero[tablero > 0]
                reward -= np.mean(np.log2(numDifCero))

        # Premio por mantener el tablero vacío
        reward += np.sum(tablero == 0)

        return reward



"""
def jugar_al_azar(n):
    continuar=True
    movs=0
    tablero=crear_tablero(n)
    tablero=llenar_pos_vacias(tablero,2)
    while continuar:
        lista=["izquierda","derecha","arriba","abajo"]
        direccion=lista[random.randint(0,3)]
        tablero=mover(tablero,direccion)
        continuar= not esta_atascado(tablero)
        movs+=1
        print(tablero)
        tablero=llenar_pos_vacias(tablero,1)
    return (movs,int(tablero.sum()),int(tablero.max()))

print(jugar_al_azar(4))
"""