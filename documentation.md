# Documentacion
---
## Directorio: `TPIA\game`

### Clase `Game()`
#### Descripcion:
En esta clase encontraremos los métodos que contemplan la 'lógica de negocio' del juego.

Metodos:

- `crear_tablero`:
    - #### Descripción
        Recibe `n` como parámetro y retorna un array numpy de tamaño `n x n` lleno de ceros.
    
    - #### Ejemplo de uso
        ```python
        print(obj.crear_tablero(4))
        # Salida
        [[0. 0. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 0. 0.]]
        ```

- `listar_pos_vacias`:
    - #### Descripción
        Recibe un `tablero` como parámetro y retorna una lista con las posiciones vacías (valor 0) dentro del tablero.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.array([[2, 0, 2], [0, 4, 0], [2, 2, 0]])
        print(obj.listar_pos_vacias(tablero))
        # Salida: [(0, 1), (1, 0), (1, 2), (2, 2)]
        ```

- `llenar_pos_vacias`:
    - #### Descripción
        Recibe un `tablero` y un `cantidad` como parámetros, y llena las posiciones vacías del tablero con el valor `2` hasta completar la cantidad indicada.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.zeros((4, 4))
        print(obj.llenar_pos_vacias(tablero, 3))
        # Salida: Un tablero con 3 posiciones llenas de 2.
        ```

- `mover_fichas_un_lugar`:
    - #### Descripción
        Recibe una fila del tablero y mueve las fichas no nulas hacia la izquierda, desplazando las fichas vacías (valor 0) hacia la derecha.
    
    - #### Ejemplo de uso
        ```python
        fila = [2, 0, 2, 4]
        print(obj.mover_fichas_un_lugar(fila))
        # Salida: [2, 2, 4, 0]
        ```

- `colapsar_fichas`:
    - #### Descripción
        Aplica la lógica de movimiento y suma de fichas a lo largo de una fila, colapsando fichas iguales adyacentes en el proceso.
    
    - #### Ejemplo de uso
        ```python
        fila = [2, 2, 4, 4]
        print(obj.colapsar_fichas(fila))
        # Salida: [4, 8, 0, 0]
        ```

- `colapsar`:
    - #### Descripción
        Aplica la función `colapsar_fichas` a cada fila del tablero. Si el tablero cambia, llena una posición vacía con un 2.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.array([[2, 2, 4, 4], [0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]])
        print(obj.colapsar(tablero))
        # Salida: Un tablero con fichas colapsadas y posiciones llenas.
        ```

- `mover`:
    - #### Descripción
        Recibe un `tablero` y una `direccion` como parámetros (puede ser "izquierda", "abajo", "derecha", "arriba"), rota el tablero según la dirección y luego aplica la función `colapsar`.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.array([[2, 2, 4, 4], [0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]])
        print(obj.mover(tablero, "izquierda"))
        # Salida: Un tablero con las fichas movidas a la izquierda.
        ```

- `esta_atascado`:
    - #### Descripción
        Recibe un `tablero` y determina si no es posible realizar más movimientos, es decir, si el juego está atascado.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.array([[2, 2, 4, 4], [4, 4, 4, 4], [8, 8, 8, 8], [16, 16, 16, 16]])
        print(obj.esta_atascado(tablero))
        # Salida: True si no hay movimientos posibles, False en caso contrario.
        ```

- `reward`:
    - #### Descripción
        Recibe un `tablero` como parámetro y retorna la recompensa, que es el valor total de las fichas en el tablero multiplicado por 10.
    
    - #### Ejemplo de uso
        ```python
        tablero = np.array([[2, 2, 4, 4], [0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]])
        print(obj.reward(tablero))
        # Salida: 80
        ```
---
### Clase `GameGui()`
#### Descripción:
Esta clase maneja la interfaz gráfica del juego utilizando la librería `tkinter`. Permite crear el tablero visual, interactuar con él, y realizar las acciones necesarias para jugar.

#### Métodos:

- `__init__`:
    - #### Descripción
        Inicializa la interfaz gráfica, configura los elementos visuales y organiza las disposiciones del tablero.
    
    - #### Ejemplo de uso
        ```python
        game_gui = GameGui()  # Crea una instancia de la interfaz gráfica
        ```

- `crear_widgets`:
    - #### Descripción
        Crea los widgets necesarios para representar el tablero del juego, incluyendo botones para cada celda.
    
    - #### Ejemplo de uso
        ```python
        game_gui.crear_widgets()  # Crea todos los botones del tablero
        ```

- `actualizar_tablero`:
    - #### Descripción
        Actualiza la interfaz gráfica del tablero con los valores actuales de las celdas (como los números en el tablero de juego).
    
    - #### Ejemplo de uso
        ```python
        game_gui.actualizar_tablero()  # Actualiza los valores en los botones con los valores del tablero
        ```

- `actualizar_texto`:
    - #### Descripción
        Actualiza el texto que muestra la interfaz gráfica, como el puntaje, el mensaje de fin de juego, etc.
    
    - #### Ejemplo de uso
        ```python
        game_gui.actualizar_texto("Juego Terminado")  # Muestra un mensaje en la interfaz
        ```

- `mover`:
    - #### Descripción
        Recibe una dirección y realiza el movimiento de las fichas en la interfaz gráfica.
    
    - #### Ejemplo de uso
        ```python
        game_gui.mover("izquierda")  # Realiza el movimiento hacia la izquierda
        ```

- `reiniciar`:
    - #### Descripción
        Reinicia el tablero, reseteando todas las celdas y comenzando un nuevo juego.
    
    - #### Ejemplo de uso
        ```python
        game_gui.reiniciar()  # Reinicia el juego
        ```

- `cambiar_tablero`:
    - #### Descripción
        Recibe un nuevo tablero y actualiza los valores de las celdas en la interfaz gráfica.
    
    - #### Ejemplo de uso
        ```python
        game_gui.cambiar_tablero(tablero_nuevo)  # Actualiza la interfaz con el nuevo tablero
        ```

- `crear_ventana`:
    - #### Descripción
        Crea la ventana principal del juego utilizando `tkinter`, donde se visualizarán todos los componentes gráficos.
    
    - #### Ejemplo de uso
        ```python
        game_gui.crear_ventana()  # Crea y configura la ventana gráfica
        ```

- `iniciar`:
    - #### Descripción
        Inicia la ejecución del juego, mostrando la interfaz gráfica y comenzando el ciclo de eventos de `tkinter`.
    
    - #### Ejemplo de uso
        ```python
        game_gui.iniciar()  # Inicia la ejecución de la interfaz gráfica
        ```

- `configurar_ventana`:
    - #### Descripción
        Configura las propiedades de la ventana principal del juego, como el tamaño, título y otros parámetros visuales.
    
    - #### Ejemplo de uso
        ```python
        game_gui.configurar_ventana()  # Configura las propiedades de la ventana
        ```
---
## Directorio: `TPIA\model`
### Clase `QNetwork`
#### Descripción:
Esta clase define una red neuronal para el aprendizaje de políticas en el contexto de un agente de refuerzo. La red se usa tanto para la política (policy) como para la red objetivo (target), ambas con la misma estructura. La red predice los valores Q (calificación de acción) para cada acción, dados los estados del entorno. Estos valores Q son usados por el agente para decidir qué acciones tomar en cada estado.

#### Métodos:

- `__init__(self, stateSize, actionSize)`:
    - #### Descripción
        El constructor de la clase, que inicializa la estructura de la red neuronal. Define tres capas `fully connected` (FC) para transformar los estados de entrada en valores Q para cada acción.
    
    - #### Argumentos
        - `stateSize`: El número de características del estado (dimensión del espacio de estados).
        - `actionSize`: El número de acciones posibles (dimensión del espacio de acciones).
    
    - #### Detalles
        - La red consta de tres capas:
          - `fc1`: Capa de entrada que toma el tamaño del estado y lo transforma a 128 unidades.
          - `fc2`: Segunda capa que toma la salida de la capa anterior y la transforma nuevamente a 128 unidades.
          - `fc3`: Capa de salida que toma los 128 valores anteriores y genera los valores Q para cada acción posible.
    
    - #### Ejemplo de uso
        ```python
        q_network = QNetwork(stateSize=10, actionSize=4)
        # La red tiene 10 entradas (dimensión del estado) y 4 salidas (acciones posibles).
        ```

- `forward(self, state)`:
    - #### Descripción
        Define el paso hacia adelante (forward pass) de la red neuronal. Toma un estado como entrada y devuelve los valores Q predichos para cada acción.
    
    - #### Argumentos
        - `state`: El estado del entorno para el cual queremos predecir los valores Q. Este debe ser un tensor de tamaño `(batch_size, stateSize)`.

    - #### Proceso
        - Primero pasa el `state` por la capa `fc1`, seguida de una activación ReLU para introducir no linealidad.
        - Luego pasa por la capa `fc2` con otra activación ReLU.
        - Finalmente, la salida pasa por la capa `fc3` y se aplica otra activación ReLU. Los valores resultantes son los valores Q para cada acción.

    - #### Ejemplo de uso
        ```python
        q_values = q_network(state)  # Calcula los valores Q para el estado dado
        ```

- `createNetwork(self, stateSize, actionSize)`:
    - #### Descripción
        Crea e inicializa dos redes neuronales: una para la política (`policyNet`) y otra para la red objetivo (`targetNet`). La red objetivo tiene el mismo inicio que la política, pero no se actualiza continuamente durante el entrenamiento. En lugar de eso, se actualiza periódicamente con un retraso respecto a los pesos de la red de política.
    
    - #### Argumentos
        - `stateSize`: El tamaño del espacio de estados (número de características del estado).
        - `actionSize`: El tamaño del espacio de acciones (número de acciones posibles).
    
    - #### Proceso
        - Se crea una instancia de `QNetwork` llamada `policyNet` que se usará para la política del agente.
        - Luego se crea otra instancia de `QNetwork` llamada `targetNet`. Se cargan los pesos de `policyNet` en `targetNet` para que inicialmente ambas redes sean iguales.
    
    - #### Detalles
        La red objetivo (`targetNet`) no se actualiza con cada paso de entrenamiento, lo que ayuda a estabilizar el aprendizaje. En su lugar, se actualiza de manera periódica (generalmente en intervalos de pasos de entrenamiento) copiando los pesos de la red de política.

    - #### Ejemplo de uso
        ```python
        policyNet, targetNet = q_network.createNetwork(stateSize=10, actionSize=4)
        # Crea dos redes: policyNet y targetNet, con 10 entradas y 4 salidas.
        ```

#### Notas adicionales:
- **Red de política (policyNet):** Esta red es la que se utiliza para predecir los valores Q durante el entrenamiento, es decir, para elegir qué acción tomar en un estado dado.
- **Red objetivo (targetNet):** Esta red tiene el mismo propósito que la red de política, pero sus pesos se actualizan con un retraso para evitar inestabilidad en el entrenamiento.
- **Activación ReLU:** Se utiliza la función de activación ReLU (Rectified Linear Unit) para introducir no linealidad en el modelo. Esto permite que la red aprenda representaciones más complejas.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Esta clase define la red neuronal que usaremos para la politica y la red objetivo.
    Ambas redes son iguales en estructura.
    """
    
    def __init__(self, stateSize, actionSize):
        """
        Inicializa la red neuronal con las capas fully connected.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(stateSize, 128)  # Capa de entrada
        self.fc2 = nn.Linear(128, 128)  # Segunda capa
        self.fc3 = nn.Linear(128, actionSize)  # Capa de salida
    
    def forward(self, state):
        """
        Define el paso hacia adelante para la red neuronal.
        Recibe un estado y devuelve los valores Q para cada acción.
        """
        x = F.relu(self.fc1(state))  # Primera capa con activación ReLU
        x = F.relu(self.fc2(x))  # Segunda capa con activación ReLU
        qValues = F.relu(self.fc3(x))  # Valores Q para cada acción
        return qValues
    
    def createNetwork(self, stateSize, actionSize):
        """
        Crea e inicializa las redes policy y target.
        Ambas redes son instancias de QNetwork, pero targetNet no se actualiza
        de manera continua durante el entrenamiento (lleva un retraso en sus pesos).
        """
        # Crear la red de política
        policyNet = QNetwork(stateSize, actionSize)
        
        # Crear la red objetivo y copiar los parámetros de la red de política
        targetNet = QNetwork(stateSize, actionSize)
        targetNet.load_state_dict(policyNet.state_dict())  # Cargamos en la targetNet los parámetros de policyNet
        
        return policyNet, targetNet

