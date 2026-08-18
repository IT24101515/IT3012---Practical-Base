import random


class SimpleReflexAgent:
    def __init__(self):
        self.actions = ['Left', 'Right', 'Up', 'Down', 'Collect', 'Stay']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Collect'
        if percept.get('wall_ahead'):
            return random.choice(['Left', 'Right', 'Up', 'Down'])
        return random.choice(['Left', 'Right', 'Up', 'Down'])


class ModelBasedAgent:
    def __init__(self):
        self.previous_percept = None
        self.previous_action = None
        self.actions = ['Left', 'Right', 'Up', 'Down']

    def sense_and_act(self, percept: dict) -> str:
        key = tuple(sorted(percept.items()))
        if key == self.previous_percept and self.previous_action is not None:
            for a in self.actions:
                if a != self.previous_action:
                    action = a
                    break
        else:
            action = random.choice(self.actions)
        self.previous_percept = key
        self.previous_action = action
        return action


class SearchAgent:
    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        from collections import deque

        wset = set(walls)
        max_x, max_y = grid_size
        moves = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        def in_bounds(pos):
            x, y = pos
            return 0 <= x < max_x and 0 <= y < max_y

        queue = deque()
        queue.append((start_pos, []))
        visited = set([start_pos])

        while queue:
            pos, path = queue.popleft()
            if pos == goal_pos:
                return path
            for name, delta in moves:
                new = (pos[0] + delta[0], pos[1] + delta[1])
                if not in_bounds(new):
                    continue
                if new in wset:
                    continue
                if new in visited:
                    continue
                visited.add(new)
                queue.append((new, path + [name]))
        return None