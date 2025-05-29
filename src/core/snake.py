import pygame
import numpy as np
import sys

class SnakeVisualizer:
    def __init__(self, grid_size, cell_size=20):
        """Initialize the visual display of the Snake game"""
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Calculate screen size in pixels
        self.screen_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

    def render(self, observation):
        """Display the current state of the Snake game"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))

        # If the observation is flat (1D), reshape it to 3D
        if len(observation.shape) == 1:
            # The AI uses 11 information channels
            num_channels = 11
            observation = observation.reshape((num_channels, self.grid_size[0], self.grid_size[1]))

        # Extract the 3 main channels for display
        body_channel = observation[0]  # Where the body segments are
        head_channel = observation[1]  # Where the head is
        apple_channel = observation[2]  # Where the apple is

        # Draw the snake BODY (in green)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if body_channel[x, y] == 1:  # If a body segment is present
                    pygame.draw.rect(
                        self.screen,
                        (0, 255, 0),  # Green color
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        # Draw the snake HEAD (in blue)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if head_channel[x, y] == 1:  # If the head is present
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 255),  # Blue color
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        # Draw the APPLE (in red)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if apple_channel[x, y] == 1:  # If an apple is present
                    pygame.draw.rect(
                        self.screen,
                        (255, 0, 0),  # Red color
                        pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                    )

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        """Close the Pygame window"""
        pygame.quit()