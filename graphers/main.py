import pygame as pg
import numpy as np
from pygame.constants import MOUSEBUTTONDOWN

class Text:
    pass

class Btn:
    pass

class Scene:

    def __init__(self, id, name, data = [], buttons = [], texts = []) -> None:
        self.id = id
        self.name = name
        self.data = data
        self.buttons = self.buttons
        self.texts = texts

    def blit (self):
        pass

class Visualizer:

    def __init__(self, screen_width, screen_height) -> None:
        pg.init()

        self.screen = pg.display.set_mode((screen_width, screen_height))

        self.running = False

        empty_scene = Scene(0, "B")

        self.scenes : list[Scene] = [empty_scene]

        self.active_scene : Scene = empty_scene

    def clean (self):
        pg.quit()
        exit()

    def addScene (self, data, btns : list[Btn] = [], texts : list[Text] = [], name = ""):
        pass

    def activate (self, i = 0, name = ""):

        scene = self.active_scene

        if name:
            selection = list(filter(lambda x: x.name == name, self.scenes))
            if len(selection) == 1:
                scene = selection[0]
        elif i < 0 or i >= len(self.scenes):
            scene = self.active_scene
        else:
            scene = self.scenes[i]

        self.active_scene = scene


    def _handleClicks (self, event : pg.event.Event):
        pass

    def draw(self):
        pass

    def mainLoop (self):

        while (self.running):

            ### manage events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                if event.type == MOUSEBUTTONDOWN:
                    self._handleClicks(event)

            ### drawing

        self.clean()
