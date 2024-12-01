import pygame
import sys
from dosmil import Game

SCREENSIZE = 600

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ZERO = (230, 230, 230)
ONE = (255, 186, 8)
TWO = (250, 163, 7)
THREE = (244, 140, 6)
FOUR = (232, 93, 4)
FIVE = (220, 47, 2)
SIX = (208, 0, 0)
SEVEN = (157, 2, 8)
EIGHT = (106, 4, 15)
NINE = (55, 6, 23)
TEN = (3, 7, 30)

COLORS = {
    0: ZERO,
    2**1: ONE,
    2**2: TWO,
    2**3: THREE,
    2**4: FOUR,
    2**5: FIVE,
    2**6: SIX,
    2**7: SEVEN,
    2**8: EIGHT,
    2**9: NINE,
    2**10: TEN
}

GAP = 10

class GameGUI:
    def __init__(self, n):
        self.game = Game() # nstanciamos el juego
        self.n = n
        self.screen = pygame.display.set_mode((SCREENSIZE, SCREENSIZE))
        self.tablero = self.game.crear_tablero(n)
        self.tablero = self.game.llenar_pos_vacias(self.tablero, 2)
        self.tile_width = (SCREENSIZE - (n + 1) * GAP) / n
        self.tile_size = (self.tile_width, self.tile_width)


    def display_message(self, message, color):
        self.screen.fill(WHITE)
        font = pygame.font.Font(None, 50)
        text = font.render(message, True, color)
        text_rect = text.get_rect(center=(SCREENSIZE // 2, SCREENSIZE // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def display_arr(self):
        self.screen.fill(WHITE)
        n, _ = self.tablero.shape
        tile_width = self.tile_size[0]
        for i in range(n):
            for j in range(n):
                x = GAP + tile_width / 2 + j * (tile_width + GAP)
                y = GAP + tile_width / 2 + i * (tile_width + GAP)
                num = int(self.tablero[(i, j)])
                font = pygame.font.Font(None, int(tile_width / len(str(num)) ** 0.3))
                color = COLORS.get(num, BLACK)
                rect = pygame.Rect(0, 0, *self.tile_size)
                rect.center = (x, y)
                bg_surface = pygame.Surface(self.tile_size)
                bg_surface.fill(color)
                self.screen.blit(bg_surface, rect)
                if num != 0:
                    text = font.render(str(num), True, BLACK if num < 128 else WHITE)
                    offset = tile_width / 2
                    text_rect = text.get_rect(center=(x, y + 0.05 * tile_width))
                    self.screen.blit(text, text_rect)

        pygame.display.flip()

    def run(self):
        pygame.init()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.tablero = self.game.mover(self.tablero, "derecha")
                    elif event.key == pygame.K_LEFT:
                        self.tablero = self.game.mover(self.tablero, "izquierda")
                    elif event.key == pygame.K_UP:
                        self.tablero = self.game.mover(self.tablero, "arriba")
                    elif event.key == pygame.K_DOWN:
                        self.tablero = self.game.mover(self.tablero, "abajo")
                    elif event.key == pygame.K_RETURN and self.esta_atascado(self.tablero):
                        self.tablero = self.game.crear_tablero(self.n)
                        self.tablero = self.game.llenar_pos_vacias(self.tablero, 2)

            if self.game.esta_atascado(self.tablero):
                reward = -1
                self.display_message("Presioná Enter para volver a jugar", RED)
            else:
                self.display_arr()


def main(n):
    game_gui = GameGUI(n)  # Creamos la instancia de la clase GameGUI
    game_gui.run()  # Ejecutamos el juego


if __name__ == "__main__":
    main(4)
