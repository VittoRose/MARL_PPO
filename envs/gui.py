import pygame as pg
import numpy as np

from coverage import GridCoverage

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
gray = (128, 128, 128)
line_width = 5

class GUI():
    """
    Class that show the GridCoverage enviroment using pygame
    """

    # TODO: When more map available update init to select the map
    def __init__(self, env):

        pg.init()

        # Get parameters from enviroment
        self.col = env.w
        self.row = env.h

        if self.col == 5:
            coef = 160
        else:
            raise NotImplementedError("Lazy dev here")

        # Offset from the screen edge
        self.offset = 20

        # Screen dimension
        self.w = coef*self.col
        self.h = coef*self.col
        self.w_tot = self.w + 2*self.offset
        self.h_tot = self.h + 100

        # Init screen
        self.screen = pg.display.set_mode((self.w_tot,self.h_tot))
        pg.display.set_caption("MARL Coverage GUI test")

        # Font 
        self.font = pg.font.SysFont('Arial', 28)
        pg.font.init()

        # Dimensions for each square
        size = (self.w//self.col, self.h//self.row)

        # Create a grid with each tile as an object
        self.grid = [[Tile( size,
                           (size[0]*i+size[0]/2+self.offset, size[1]*j+size[1]/2 + self.offset),
                           self.screen, 
                           self.font,
                           (i,j)
                           ) 
                        for i in range(self.col)] 
                        for j in range(self.row)]

        # Select tiles that are obstacles
        for i in range(self.row):
            for j in range(self.col):
                self.grid[i][j].am_i_obstacle(env.grid[i,j])

    def update(self, env):
        """
        Get the value from GridCoverage and display on the gui
        """
        for i in range(self.row):
            for j in range(self.col):
                self.grid[i][j].state(env.grid[i,j])

        self.draw(env)

    def draw(self, env):
        """
        Draw all the elements on the screen
        """

        self.screen.fill(white)
        
        for i in range(self.row): 
            for j in range(self.col):
                self.grid[i][j].draw(env)

        # Draw missing border
        pg.draw.rect(self.screen, black, (self.offset+self.w, self.offset, line_width, self.h+line_width))
        pg.draw.rect(self.screen, black, (self.offset, self.offset+self.h, self.w, line_width))

        pg.display.flip()
        pg.time.Clock().tick(15)

class Tile():
    """
    Class that represent a single tile on the grid
    :param size: (width, heigth) in pixel
    :param center: (x,y) in pixel from the top left corner of the screen
    :param screen: pygame screen
    :param font: font for agent id display
    :param index: (i,j) index to identify the tile in the grid 
    """
    def __init__(self, size: tuple[int, int], center: tuple[int, int], screen, font, index: tuple[int,int]):
        
        self.size = size
        self.center = center
        self.screen = screen
        self.font = font

        self.top_right = (center[0] - size[0]/2, center[1] - size[1]/2)

        self.i = index[0]
        self.j = index[1]

        self.clean = None
        self.obstacle = None

    def am_i_obstacle(self, flag) -> None:
        """
        Get the value of a grid element, and if it's a obstacle change a flag for drawing later
        Used once in the creation of the grid
        """
        if flag == -1:
            self.obstacle = True

    def state(self, agent) -> None:
        """
        Check if the tile si covered by one angent, update every .step()
        """
        if self.obstacle is not True:
            self.clean = agent

    def draw(self, env) -> None:
        """
        Draw upper and left side of the square
        Fill the square with the required color if needed
        Write the id of the agent on the tile if somone is there
        :param env: GridCoverage instance

        """

        # Draw upper and left line of the tile
        pg.draw.rect(self.screen, black, (self.top_right[0], self.top_right[1], self.size[0], line_width))
        pg.draw.rect(self.screen, black, (self.top_right[0], self.top_right[1], line_width, self.size[1]))

        # Fill the tile with the required color
        if self.obstacle:
            pg.draw.rect(self.screen, gray, (self.top_right[0]+line_width, self.top_right[1]+line_width, self.size[0]-line_width, self.size[1]-line_width))

        if self.clean == 1:
            pg.draw.rect(self.screen, red, (self.top_right[0]+line_width, self.top_right[1]+line_width, self.size[0]-line_width, self.size[1]-line_width))
        elif self.clean == 2:
            pg.draw.rect(self.screen, green, (self.top_right[0]+line_width, self.top_right[1]+line_width, self.size[0]-line_width, self.size[1]-line_width))
        
        id = self.somone_here(env)

        if id == 0:
            text = self.font.render("0", True, black)
            self.screen.blit(text, self.center)
        elif id == 1:
            text = self.font.render("1", True, black)
            self.screen.blit(text, self.center)
    
    def somone_here(self, env):
        """
        Check if an agent is in this tile,
        :param env: GridCoverage instance
            return the agent id if is here
            return None otherwise
        """
        id = None

        for indx, agent in enumerate(env.agent_xy):
            if agent[0] == self.j and agent[1] == self.i:
                print(f"Agent {indx} is in ({self.i},{self.j}) as expected {agent}")
                id = indx
                break       # Only one agent can be here, no need to continue
        return id
    








if __name__ == "__main__":
    import time

    env = GridCoverage(1)
    env.reset()
    screen = GUI(env)

    
    run = True

    t0 = time.time()

    while run:
        time.sleep(2)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        _, reward, _, _, _ =  env.step([np.random.randint(5), np.random.randint(5)])
        screen.update(env)
