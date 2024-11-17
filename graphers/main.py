import pygame as pg
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Btn:
    pass
@dataclass
class Text:
    pass

@dataclass
class Scene:

    iid : int
    name : str
    data : np.ndarray 
    btns : list[Btn] = field(default_factory=list)
    texts : list[Text] = field(default_factory=list)


    def blit (self, surface):
        pass

class Visualizer:

    def __init__(self, screen_width, screen_height) -> None:
        pg.init()

        self.screen = pg.display.set_mode((screen_width, screen_height))

        self.surface_screen = pg.Surface()

        self.running = False

        empty_scene = Scene(0, "B", data=np.array([]))

        self.scenes : list[Scene] = [empty_scene]

        self.active_scene : Scene = empty_scene

    def clean (self):
        pg.quit()
        exit()

    def addScene (self, data, btns : list[Btn] = [], texts : list[Text] = [], name = ""):
        self.scenes.append(Scene(len(self.scenes), name, data, btns, texts))

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

                if event.type == pg.MOUSEBUTTONDOWN:
                    self._handleClicks(event)

            ### drawing
            self.active_scene.blit(self.screen)
        self.clean()
