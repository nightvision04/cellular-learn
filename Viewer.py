import time
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np
import cv2
from Genotype import JSONData


class Playback():
    '''
    Displays and refreshes the task using the task properties:
    '''

    def __init__(self,data_array=[],text='',dna=[]):

        assert data_array, 'data_array was empty'
        #assert dna, 'data_array was empty'
        self.data_array = data_array
        self.dna = dna.T
        self.text = text

    def run(self):
        '''
        Display frame by frame of self.data_array (RGB array)
        '''

        pygame.init()
        scale_xy = 6

        # Screen will be 2x2 lalyer matrices
        screen = pygame.display.set_mode((self.data_array[-1][0].shape[0]*scale_xy*2,
                                          self.data_array[-1][0].shape[1]*scale_xy*2))
        pygame.display.set_caption("Cellular-Learn")
        icon = pygame.image.load('cellular-learn.png')
        clock = pygame.time.Clock()

        running = True
        max_loops =1
        loop=0
        i=0
        while running:
            clock.tick()

            # Process all events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    exit()

            # screen.fill((0, 0, 0))

            # Blit current world
            topStack = np.hstack((self.data_array[i][0],self.data_array[i][1]))
            bottomStack = np.hstack((self.data_array[i][2], self.data_array[i][3]))
            img = np.vstack((topStack,bottomStack))

            src = cv2.resize(img,
                             dsize=(img.shape[1]*scale_xy,img.shape[0]*scale_xy),
                             interpolation=cv2.INTER_NEAREST)
            pygame.surfarray.blit_array(screen,src)

            # Blit dna
            colors = [[0, 0, 0],
                      [255,255,255],
                      [34,255,200],
                      [3,2,255],
                      [100,2,255],
                      [230,0,0],
                      [0, 0,90],
                      [255, 0, 255],
                      [34, 40, 200],
                      [94, 2, 30],
                      [100, 200, 255],
                      [230, 0, 80],
                      ]
            dna_rgb = np.ones((self.dna.shape[0],self.dna.shape[1],3))
            for x in range(self.dna.shape[0]):
                for y in range(self.dna.shape[1]):
                    dna_rgb[x][y] = colors[self.dna[x][y]]
            surface = pygame.surfarray.make_surface(dna_rgb)
            surface = pygame.transform.scale(surface, (self.dna.shape[0], self.dna.shape[1]))  # Scaled a bit.
            screen.blit(surface, (0, 1))


            # # Display fps
            font = pygame.font.Font(None, 20)
            # fps = font.render('FPS:'+str(int(clock.get_fps())), True, pygame.Color('red'))
            # screen.blit(fps, (10, 10))
            #
            # Display Gen
            text = font.render(self.text, True, pygame.Color('orange'))
            screen.blit(text, (self.dna.shape[0]+5, 1))

            # Update display
            pygame.display.update()

            if i+1 == len(self.data_array):
                loop+=1
                i=0

            if loop==max_loops:
                running = False

            i+=1

        return
