# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.distance_calculator import Distancer


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):

        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
       
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a, food_left) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        walls = game_state.get_walls()
        x2 = walls.width // 2
        y2 = walls.height // 2
        if self.red:
            xboundary = x2 - 1
        else:
            xboundary = x2

        if self.get_agent_state(self.index).num_carrying > 3:
            print(self.get_agent_state(self.index).num_carrying)
            best_dist = 9999
            best_action = None
            for action in actions:
                dist = self.get_maze_distance((xboundary, y2), self.get_successor(game_state, action).get_agent_position(self.index))
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        if food_left<= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
    
    def evaluate(self, game_state, action, food_left):
        features = self.get_features(game_state, action, food_left)
        weights = self.get_weights(game_state, action, food_left)
        return features * weights


    def get_features(self, game_state, action, food_left):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

            my_state = successor.get_agent_state(self.index)
            my_pos = my_state.get_position()
            if food_left <= 19:
                
                features = util.Counter()
                successor = self.get_successor(game_state, action)

                enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
                features['num_invaders'] = len(invaders)
                if len(invaders) > 0:
                    dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
                    features['invader_distance'] = min(dists)
                
                if action == Directions.STOP: 
                    features['stop'] = 1
                rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
                print("\n  1.1  action=", action)
                if action == rev: 
                    features['reverse'] = 1
                    print("\n  1.2  action=", action)

        return features

    def get_weights(self, game_state, action, food_left):
        if food_left<= 19:
            return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
        
        return {'successor_score': 100, 'distance_to_food': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):

        enemies = self.get_opponents(game_state)
        for enemy in enemies:
            state = game_state.get_agent_state(enemy)
            if state.is_pacman and state.get_position() is not None:
                enemies.append(state.get_position())

        if enemies:
            enemy = min(enemies, key=lambda inv: self.distancer.get_distance(self.get_current_position(game_state), inv))
            action = self.ab_search(game_state, depth=2, target=enemy)
            if action:
                return action
        else:
            walls = game_state.get_walls()
            x2 = walls.width // 2
            y2 = walls.height // 2
            if self.red:
                xboundary = x2 - 1
            else:
                xboundary = x2
            action = self.ab_search(game_state, depth=2, target=(xboundary, y2))
            return action
    def ab_search(self, game_state, depth, target=None):
        def max_value(state, depth, alpha, beta):
            if depth == 0 or state.is_over():
                return self.evaluate(state, target)
            value = -9999
            for action in state.get_legal_actions(self.index):
                successor = state.generate_successor(self.index, action)
                value = max(value, min_value(successor, depth -1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state, depth, alpha, beta):
            if depth ==0 or state.is_over():
                return self.evaluate(state, target)
            value = 9999
            opponents = self.get_opponents(state)
            for opponent in opponents:
                opponent_state = state.get_agent_state(opponent)
                if opponent_state is None or opponent_state.get_position() is None:
                    continue
                for action in state.get_legal_actions(opponent):
                    successor = state.generate_successor(opponent, action)
                    value = min(value, max_value(successor, depth -1, alpha, beta))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
            return value

        best_action = None
        best_score = -9999
        alpha = -9999
        beta = 9999

        for action in game_state.get_legal_actions(self.index):
            successor = game_state.generate_successor(self.index, action)
            score = min_value(successor, depth -1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        return best_action

    def evaluate(self, game_state, target):
        pos = self.get_current_position(game_state)
        enemies = self.get_opponents(game_state)
        for enemy in enemies:
            state = game_state.get_agent_state(enemy)
            if state.is_pacman and state.get_position() is not None:
                enemies.append(state.get_position())
        score = -100 * len(enemies)

        return score