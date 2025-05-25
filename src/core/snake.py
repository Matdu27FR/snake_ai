import pygame
import numpy as np
import sys

class SnakeVisualizer:
    def __init__(self, grid_size, cell_size=20):
        """Initialise l'affichage visuel du jeu Snake"""
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Calcul de la taille de l'écran en pixels
        self.screen_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        
        # Initialisation de Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

    def render(self, observation):
        """Affiche l'état actuel du jeu Snake"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))

        # Si l'observation est à plat (1D), la reformater en 3D
        if len(observation.shape) == 1:
            # L'IA utilise 11 canaux d'informations
            num_channels = 11
            observation = observation.reshape((num_channels, self.grid_size[0], self.grid_size[1]))

        # On extrait les 3 canaux principaux pour l'affichage
        body_channel = observation[0]  # Où sont les segments du corps
        head_channel = observation[1]  # Où est la tête
        apple_channel = observation[2]  # Où est la pomme

        # Dessiner le CORPS du serpent (en vert)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if body_channel[x, y] == 1:  # Si un segment de corps est présent
                    pygame.draw.rect(
                        self.screen,
                        (0, 255, 0),  # Couleur verte
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        # Dessiner la TÊTE du serpent (en bleu)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if head_channel[x, y] == 1:  # Si la tête est présente
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 255),  # Couleur bleue
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        # Dessiner la POMME (en rouge)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if apple_channel[x, y] == 1:  # Si une pomme est présente
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),  # Couleur rouge
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        """Ferme la fenêtre Pygame"""
        pygame.quit()