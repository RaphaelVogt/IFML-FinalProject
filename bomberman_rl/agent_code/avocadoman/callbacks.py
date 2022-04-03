import os
import pickle
import random

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup the code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if not os.path.isfile("avocadoman-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))
        self.is_fit = False

    else:
        self.logger.info("Loading model from saved state.")
        with open("avocadoman-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.is_fit = True

    self.epsilon = .3
    
def act(self, game_state: dict) -> str:
    """
    The agent parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if (self.train and random.random() < self.epsilon) or self.is_fit == False:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    features = state_to_features(game_state)
    self.logger.info(f"Feature vector: {features}")
    action = ACTIONS[np.argmax(self.model.predict(features.reshape(1, -1)))]
    self.logger.info(f"Choose action {action}")
    return action

def look_for_targets(free_space, start, targets, logger=None, mode="NOT_SAVE"):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if targets is None or len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1    

    good_choice = False
    nearbies = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])
    if mode == "SAVE":
        for target in targets:
            if target in nearbies + np.array(best):   # is target next to best position
                good_choice = True
                break

        if not good_choice:
            better_choices = []
            for key in parent_dict:     # safe all better choices
                for target in targets:
                    if target in nearbies + np.array(key):
                        better_choices.append(np.array([key[0], key[1]]))
                        break

            if len(better_choices) != 0:    # better choice found
                distances = np.linalg.norm(np.array(better_choices) - np.array(start), axis=1)  # euclidean distance
                best = tuple(better_choices[np.argmin(distances)])


    current = best
    path = []
    while True:
        path.insert(0, current)
        if parent_dict[current] == start:
            return path
        current = parent_dict[current]

def look_for_targets_strict(free_space, start, targets, logger=None):
    '''
    Similar to look_for_targets but it only return the path to the target
    if the target is actually reachable. Otherwise return None.
    '''

    best_path = look_for_targets(free_space, start, targets)
    if best_path is not None and (len(best_path) != 0) and (best_path[-1] in targets):
        return best_path
    else:
        return None

def get_blast_radius(bombs, field):
    '''
    A function taht returns a list of all tiles on the field
    whitch are currently occupied by a bomb and its blast radius.
    '''
    
    # Initialize an array that stores the positions of all tiles 
    # that are within a bomb's blast radius
    blast_radius = []
    bombs_size = len(bombs)

    if bombs_size != 0:
        for i in range(bombs_size):
            # Append the positions of the bombs
            blast_radius.append(bombs[i]) 
            # Initialize x and y with the coordinates of the bombs
            x = bombs[i][0]
            y = bombs[i][1]
            # Check if the explosion propagation to the right is blocked by a wall
            for j in range(1,4): 
                explosion_propagation_to_the_right = x+j,y
                if explosion_propagation_to_the_right in field and field[explosion_propagation_to_the_right] == -1: break
                blast_radius.append(explosion_propagation_to_the_right)
            # Check if the explosion propagation to the left is blocked by a wall
            for j in range(1,4):
                explosion_propagation_to_the_left = x-j,y
                if explosion_propagation_to_the_left in field and field[explosion_propagation_to_the_left] == -1: break
                blast_radius.append(explosion_propagation_to_the_left)
            # Check if the upward propagation of the explosion is blocked by a wall
            for j in range(1,4):
                explosion_propagation_upwards =  x,y-j
                if explosion_propagation_upwards in field and field[explosion_propagation_upwards] == -1: break
                blast_radius.append(explosion_propagation_upwards)
            # Check if the downward propagation of the explosion is blocked by a wall
            for j in range(1,4):
                explosion_propagation_downwards =  x,y+j
                if explosion_propagation_downwards in field and field[explosion_propagation_downwards] == -1: break
                blast_radius.append(explosion_propagation_downwards) 
    return blast_radius

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of your model into a feature vector.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Initialize unsefull informations about the game state
    _, _, bomb_left, (x,y) = game_state['self']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    current_position = (x,y)
    field = game_state['field']
    current_position = (x,y)
    free_space = field == 0  

    # Get positions of the other agents
    others_xy = [xy for (_, _, _, xy) in game_state['others']]
   
    # Get all bomb possitions
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, _) in bombs]

    # Mark bombs as -3 on the field
    for bomb in bomb_xys:
        field[bomb] = -3

    # Mark other players as -2 on the field
    for other in others_xy:
        field[other] = -2
    
    # Shows the countdown until the bombs explode
    bomb_map = np.ones(field.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t+1)

    blast_radius = get_blast_radius(bomb_xys, free_space)

    #########################################################
    #---------------- FEATURE: ESCAPE_MOVE -----------------#
    #########################################################
    '''
    Feature to find the next nearest escape rounte out of danger.
    If the agent is currently in danger the feature tells him
    where to go to get out of the blast raduis of a bomb but
    only if there is really an escape route and this next move
    will not be in the range in a blast radius of a bomb otherwise
    the agent is supposed to wait.
    '''
        
    escape_move_xy = np.zeros(2, dtype=int)
    safe_zone = [(x, y) for x in range(1, field.shape[0] -1) for y in range(1, field.shape[0] -1) if (field[x, y] == 0) and (x, y) not in blast_radius]
    escape_move = look_for_targets_strict(free_space, current_position, safe_zone)

    if ((escape_move is not None) and 
        (bomb_map[escape_move[0]] > 0) and 
        (explosion_map[escape_move[0]] == 0)):
        escape_move_xy[0] = escape_move[0][0] - x
        escape_move_xy[1] = escape_move[0][1] - y   

    #########################################################
    #---------------- FEATURE: NEXT_COIN -------------------#
    #########################################################
    '''
    Feature to find the next nearest coin.
    If there are currently coins revealed on the field the agent
    is toled go collect them but if it is not in danger, the 
    next move is not in the blast radius of a bomb and only if
    he closest to the coin of all the other agents.
    '''
    
    coin_xy = np.zeros(2, dtype=int)
    path_nearest_coins = look_for_targets_strict(free_space, current_position, coins)

    distance_nearest_coin_enemy = 100     

    if len(others_xy) != 0 and path_nearest_coins is not None:
        paths = look_for_targets_strict(free_space, path_nearest_coins[-1], others_xy)
        if paths is not None:
            distance_nearest_coin_enemy = len(paths)

    if ((path_nearest_coins is not None) and
        (len(path_nearest_coins) + 1 <=  distance_nearest_coin_enemy) and
        (escape_move_xy[0] == 0 and escape_move_xy[1] == 0) and 
        (path_nearest_coins[0] not in blast_radius) and
        (explosion_map[path_nearest_coins[0]] == 0)): 
        coin_xy[0] = path_nearest_coins[0][0] - x
        coin_xy[1] = path_nearest_coins[0][1] - y
    else: 
        path_nearest_coins = None
  
    #########################################################
    #---------------- FEATURE: NEXT_CRATE ------------------#
    #########################################################
    '''
    Feature to find the next nearest crate.
    If the are crates on the field the agent will got to them
    but only if hes not in danger, no coins are reveald and 
    the next move do not bring the agent into a blast
    radius of a bomb.
    '''
    
    crate_xy = np.zeros(2, dtype=int)
    crates = [(x, y) for x in range(1, field.shape[0] -1) for y in range(1, field.shape[0] -1) if (field[x, y] == 1)]
    paths_nearest_crates = look_for_targets(free_space, current_position, crates, mode="SAVE")
  
    if ((paths_nearest_crates is not None) and  
        (escape_move_xy[0] == 0 and escape_move_xy[1] == 0) and 
        (explosion_map[paths_nearest_crates[0]] == 0) and 
        (paths_nearest_crates[0] not in blast_radius) and 
        (path_nearest_coins is None)):
        crate_xy[0] = paths_nearest_crates[0][0] - x
        crate_xy[1] = paths_nearest_crates[0][1] - y
        if crate_xy[0] == 0 and crate_xy[1] == 0:
            crate_xy[0] = 2
            crate_xy[1] = 2

    #########################################################
    #---------------- FEATURE: ENEMY -----------------------#
    #########################################################
    '''
    Feature to find the next nearest enemy.
    If there are enemys on the field and no crates and no coins 
    are left the agent, the next best move towards the enemy is
    not within a blast radius nor the agent is currently in danger,
    the agent should hunt for the other .
    '''

    enemy_xy = np.zeros(2, dtype=int)
    path_nearest_enemy = look_for_targets(free_space, current_position, others_xy, mode="SAVE")
    
    if ((path_nearest_enemy is not None) and
        (escape_move_xy[0] == 0 and escape_move_xy[1] == 0) and 
        (explosion_map[path_nearest_enemy[0]] == 0) and 
        (path_nearest_enemy[0] not in blast_radius) and 
        (bomb_map[path_nearest_enemy[0]] > 0) and
        (path_nearest_coins is None) and 
        (paths_nearest_crates is None)):
        enemy_xy[0] = path_nearest_enemy[0][0] - x
        enemy_xy[1] = path_nearest_enemy[0][1] - y

    #########################################################
    #---------------- FEATURE: ATTACK ----------------------#
    #########################################################
    '''
    Feature to tell the agent when it is supposed to place a bomb.
    If the agent is next to a crate or within a range of 2 steps
    to the next enemy the feature is set to 1 and the agent will 
    place a bomb at its position but only if it is currently not 
    in danger, no coins are left to collect, his next move towards 
    the enemy is not in the blast radius of a bomb and he has a 
    way to escape from is position.
    '''

    enemy_in_range = np.zeros(1, dtype=int)
    possible_safe_zone = [(x, y) for x in range(1, field.shape[0] -1) for y in range(1, field.shape[0] -1) if (field[x, y] == 0) and (x, y) not in get_blast_radius([current_position], field)]
    possible_escape_move = look_for_targets(free_space, current_position, possible_safe_zone)
    
    distance_to_nearest_enemy = 100

    if len(others_xy) != 0:
        path = look_for_targets_strict(free_space, current_position, others_xy)
        if path is not None:
            distance_to_nearest_enemy = len(path)

    if (((distance_to_nearest_enemy <= 2) or (crate_xy[0] == 2 and crate_xy[1] == 2)) and
        (escape_move_xy[0] == 0 and escape_move_xy[1] == 0) and 
        (possible_escape_move is not None) and 
        (explosion_map[possible_escape_move[0]] == 0) and
        (bomb_map[possible_escape_move[0]] > 0) and
        (path_nearest_coins is None) and
        (bomb_left)):
            enemy_in_range[0] = 1

    #########################################################
    #---------------- FEATURE: VECTOR ----------------------#
    #########################################################

    # Combine all the features to a feature vector
    features = np.hstack((enemy_in_range,escape_move_xy,enemy_xy,coin_xy,crate_xy))

    return features
