class game:
    def __init__(self, width, height) -> None:
        """Initialize the Snake game with a grid of specified dimensions"""
        self.width = width
        self.height = height
        self.grid = []
        self.snake = []
        self.snakehead = []
        self.food = None
        self.game_over = False
        self.win = False

    def init_grid(self):
        """Create an empty grid filled with zeros"""
        self.grid = []
        for ligne in range(self.height):
            self.grid.append([])
            for colonne in range(self.width):
                self.grid[ligne].append(0)

    def clear_grid(self):
        """Reset the grid"""
        self.grid = []
        self.init_grid()

    def print_grid(self):
        """Display the grid in the console"""
        for ligne in range(len(self.grid)):
            print(self.grid[ligne])

    def init_snake(self):
        """Initialize the snake at the center of the grid"""
        firstx = int(self.width / 2)
        firsty = int(self.height / 2)
        self.snakehead = [firstx, firsty]
        
        # Create a snake with 2 body segments
        self.snake = [
            [firstx - 1, firsty],
            [firstx - 2, firsty]
        ]
        
        self.game_over = False
        self.win = False
        self.place_food()

    def update_snake(self):
        """Update the snake display on the grid"""
        self.clear_grid()
        
        # Head = 2, Body = 1, Food = 3
        headx, heady = self.snakehead
        self.grid[heady][headx] = 2
        
        for segment in range(len(self.snake)):
            posx = self.snake[segment][0]
            posy = self.snake[segment][1]
            self.grid[posy][posx] = 1
        
        if self.food:
            foodx, foody = self.food
            self.grid[foody][foodx] = 3

    def place_food(self):
        """Place food randomly on an empty cell"""
        import random
        
        free_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if [x, y] != self.snakehead and [x, y] not in self.snake:
                    free_positions.append([x, y])
        
        if free_positions:
            self.food = random.choice(free_positions)
        else:
            self.food = None
            self.win = True
            self.game_over = True

    def check_collision(self, next_x, next_y):
        """Check if a position would cause a collision"""
        # Collision with walls or body
        if next_x < 0 or next_x >= self.width or next_y < 0 or next_y >= self.height:
            return True
        
        if [next_x, next_y] in self.snake:
            return True
        
        return False

    def direction_correct(self, direction):
        """Check if the direction is valid (0: up, 1: down, 2: left, 3: right)"""
        list_direction = [0, 1, 2, 3]
        if direction in list_direction:
            return True
        return False
    
    def move(self, direction):
        """Move the snake in the specified direction"""
        if self.game_over:
            return
        
        if self.direction_correct(direction):
            next_x, next_y = self.snakehead.copy()
            
            # 0: up, 1: down, 2: left, 3: right
            if direction == 0:
                next_y -= 1
            elif direction == 1:
                next_y += 1
            elif direction == 2:
                next_x -= 1
            elif direction == 3:
                next_x += 1
            
            if self.check_collision(next_x, next_y):
                self.game_over = True
                return
            
            eat_food = (self.food and next_x == self.food[0] and next_y == self.food[1])
            
            self.snake.insert(0, self.snakehead.copy())
            
            if not eat_food and len(self.snake) > 0:
                self.snake.pop()
                
            self.snakehead = [next_x, next_y]
            
            if eat_food:
                self.food = None
                self.place_food()
                
            self.update_snake()
        else:
            self.clear_grid()
            self.init_snake()
            self.update_snake()

if __name__ == "__main__":
    g = game(10, 10)
    g.init_grid()
    g.init_snake()
    g.update_snake()
    g.print_grid()