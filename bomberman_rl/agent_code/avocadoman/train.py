from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

import numpy as np
import time

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters
TRANSITION_HISTORY_SIZE = 4000
GAMMA = 0.99
ALPHA = 0.001

# Events
MOVE_RIGHT_DIRECTION = "MOVE_RIGHT_DIRECTION"
BAD_BOMB_SPOT = "BAD BOMB SPOT"
NOT_MOVE_RIGHT_DIRECTION = "NOT_MOVE_RIGHT_DIRECTION"
GOOD_BOMB_SPOT = "GOOD_BOMB_SPOT"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.batch_size = 50 
    self.rewardscore = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is not None:

        features_old = state_to_features(old_game_state)
        features_new = state_to_features(new_game_state)

        events = add_custom_events(events, new_game_state, old_game_state)
        reward = reward_from_events(self, events)
        
        self.transitions.append(Transition(features_old, self_action, features_new, reward))
        self.rewardscore = np.add(self.rewardscore, reward)

def add_custom_events(events, new_game_state, old_game_state):
    if old_game_state is not None:
        current_position = new_game_state['self'][3]
        old_position = old_game_state['self'][3]
        features_old = state_to_features(old_game_state)
        others_lenght = len(old_game_state['others'])
       
        enemy_in_range = features_old[0]    
        escape_move = features_old[1:3]     
        enemy =  features_old[3:5]          
        coin = features_old[5:7]            
        crate = features_old[7:9]          

        if  (escape_move[0] != 0 or escape_move[1] != 0) and current_position == (old_position[0] + escape_move[0], old_position[1] + escape_move[1]):    
            events.append(MOVE_RIGHT_DIRECTION)     # Move out of danger
        elif (escape_move[0] == 0 and escape_move[1] == 0)  and (coin[0] != 0 or coin[1] != 0) and current_position == (old_position[0] + coin[0], old_position[1] + coin[1]):
            events.append(MOVE_RIGHT_DIRECTION)     # Move to nearest coin
        elif ((escape_move[0] == 0 and escape_move[1] == 0)  and 
            (crate[0] != 0 or crate[1] != 0) and 
            (coin[0] == 0 and coin[1] == 0) and 
            (current_position == (old_position[0] + crate[0], old_position[1] + crate[1]))):
            events.append(MOVE_RIGHT_DIRECTION)     # Move to nearest crate
        elif (others_lenght > 0 and 
            (escape_move[0] == 0 and escape_move[1] == 0) and 
            (enemy[0] != 0 or enemy[1] != 0) and 
            (coin[0] == 0 and coin[1] == 0) and 
            (crate[0] == 0 and crate[1] == 0) and
            (current_position == (old_position[0] + enemy[0], old_position[1] + enemy[1]))):
            events.append(MOVE_RIGHT_DIRECTION)     # Move to nearest enemy
        
        elif (old_position == current_position and
            (escape_move[0] == 0 and escape_move[1] == 0)  == 0 and 
            enemy_in_range == 0 and 
            (enemy[0] == 0 and enemy[1] == 0) and 
            (coin[0]  == 0 and coin[1]  == 0) and 
            (crate[0] == 0 and crate[1] == 0) and 
            (escape_move[0] == 0 and escape_move[1] == 0) and'WAITED' in events):
            events.append(MOVE_RIGHT_DIRECTION)
        else:
            events.append(NOT_MOVE_RIGHT_DIRECTION)

        if enemy_in_range == 1 and 'BOMB_DROPPED' in events:
            events.append(GOOD_BOMB_SPOT)
        elif enemy_in_range == 1 and 'BOMB_DROPPED' not in events:
            events.append(BAD_BOMB_SPOT)
        
    return events


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))
    self.rewardscore = np.add(self.rewardscore, reward)
    state_matrix = np.zeros((TRANSITION_HISTORY_SIZE, len(state_to_features(last_game_state))))
    pred_matrix = np.zeros((TRANSITION_HISTORY_SIZE, 6))

    """
    For our training alogrithm we used temporal difference Q-leanring
        Q_new(s_t,a_t) <- Q(s_t,a_t) + alpha * (reward + gamma * maxQ(s_t+1,a) - Q(s_t,a_t))
    to update our model.
    """
    if last_game_state['round'] < 10 or last_game_state['round'] % 10 == 0:
        self.batch_size = min(TRANSITION_HISTORY_SIZE,self.batch_size)
        batch_indices = np.random.choice(np.arange(len(self.transitions)), size=self.batch_size)
        self.batch_size += 100
        for i, j in enumerate(batch_indices):
            state, action, state_next, reward = self.transitions[j]
            if not self.is_fit:
                new_pred = np.zeros(6)
            elif state_next is not None and state is not None:
                new_pred = self.model.predict([state])[0]
                TD = ALPHA * (reward + (GAMMA * np.max(self.model.predict([state_next])[0])-(self.model.predict([state])[0][ACTIONS.index(action)])))
                new_pred[ACTIONS.index(action)] += TD
                state_matrix[i] = state
                pred_matrix[i] = new_pred 

        self.model.fit(state_matrix, pred_matrix)
        self.is_fit = True
        self.epsilon *= .95
        self.logger.info(f"Current epsilon: {self.epsilon}.")

        # Store the model
        with open("avocadoman-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        self.logger.info("Model saved.")

    # Store how many coins agent gained after each game
    with open("improvement_coins.txt", "a") as f:
        f.write(str(last_game_state['self'][1]) + "\n")

    # Score the rewards after each game
    with open("improvement_reward.txt", "a") as f:
        f.write(str(self.rewardscore) + "\n")
        
    self.rewardscore = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -1,
        e.COIN_COLLECTED: 100,
        e.INVALID_ACTION: -5,
        MOVE_RIGHT_DIRECTION: 30, 
        NOT_MOVE_RIGHT_DIRECTION: -5,
        GOOD_BOMB_SPOT: 20,
        BAD_BOMB_SPOT: -20,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

