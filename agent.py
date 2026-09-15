# agent.py
from collections import deque
import heapq
import random
import math


class SimpleReflexAgent:
    """A simple reflex agent reacting purely to immediate percepts."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here') or percept.get('smells_food'):
            return 'Up'
        elif percept.get('wall_ahead') or percept.get('hit_wall'):
            return 'Left'
        else:
            return 'Up'


class ModelBasedAgent:
    """A model-based agent maintaining internal state/memory to prevent loops."""

    def __init__(self):
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here') or percept.get('smells_food'):
            self.last_action = 'Up'
            return 'Up'
        elif percept.get('wall_ahead') or percept.get('hit_wall'):
            if self.last_action == 'Left':
                action = 'Right'
            elif self.last_action == 'Right':
                action = 'Down'
            elif self.last_action == 'Down':
                action = 'Up'
            else:
                action = 'Left'
            self.last_action = action
            return action
        else:
            self.last_action = 'Up'
            return 'Up'


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SearchAgent:
    """Problem-solving agent implementing Uninformed and Informed Graph Search algorithms."""

    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

    def sense_and_act(self, percept: dict) -> str:
        """
        Formulates a multi-step plan to the closest food pellet if plan is empty,
        and executes the planned actions step-by-step.
        """
        if not self.plan:
            start_pos = tuple(percept.get('agent_pos', (0, 0)))
            all_food = percept.get('all_food', [])
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size', None)

            if not all_food:
                return 'Up'

            # Find closest food pellet using Manhattan distance
            closest_food = min(
                all_food,
                key=lambda food: abs(food[0] - start_pos[0]) + abs(food[1] - start_pos[1])
            )

            # Execute search method matching active_algo
            if self.active_algo == 'BFS':
                path = self.bfs_search(start_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(start_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(start_pos, closest_food, walls, grid_size)
            elif self.active_algo == 'AStar':
                path = self.astar_search(start_pos, closest_food, walls, grid_size)
            else:
                path = self.bfs_search(start_pos, closest_food, walls, grid_size)

            if path:
                self.plan = path
            else:
                return 'Up'

        if self.plan:
            return self.plan.pop(0)

        return 'Up'

    def bfs_search(self, start_pos, goal_pos, walls, grid_size=None):
        """
        Breadth-First Search (BFS) algorithm.
        Explores the shallowest nodes first using a FIFO queue (deque.popleft()).
        Maintains a reached set to track explored states (Graph Search).
        """
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls_set = {tuple(w) for w in walls}

        if start == goal:
            return []

        queue = deque([(start, [])])
        reached = {start}

        directions = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while queue:
            curr_pos, path = queue.popleft()

            for action, (dx, dy) in directions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)

                if grid_size is not None:
                    w, h = grid_size
                    if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
                        continue

                if next_pos in walls_set or next_pos in reached:
                    continue

                if next_pos == goal:
                    return path + [action]

                reached.add(next_pos)
                queue.append((next_pos, path + [action]))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size=None):
        """
        Depth-First Search (DFS) algorithm.
        Explores the deepest nodes first using a LIFO stack (list.pop()).
        Maintains a reached set to track explored states (Graph Search).
        """
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls_set = {tuple(w) for w in walls}

        if start == goal:
            return []

        stack = [(start, [])]
        reached = set()

        directions = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while stack:
            curr_pos, path = stack.pop()

            if curr_pos == goal:
                return path

            if curr_pos in reached:
                continue

            reached.add(curr_pos)

            for action, (dx, dy) in directions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)

                if grid_size is not None:
                    w, h = grid_size
                    if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
                        continue

                if next_pos in walls_set or next_pos in reached:
                    continue

                stack.append((next_pos, path + [action]))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size=None, step_costs=None):
        """
        Uniform-Cost Search (UCS) algorithm.
        Explores nodes ordered by total path cost g(n) using a Priority Queue (heapq.heappop()).
        Maintains a reached map to track minimum path cost to explored states (Graph Search).
        """
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls_set = {tuple(w) for w in walls}

        if start == goal:
            return []

        counter = 0
        pq = [(0, counter, start, [])]
        reached = {start: 0}

        directions = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while pq:
            cost, _, curr_pos, path = heapq.heappop(pq)

            if curr_pos == goal:
                return path

            if cost > reached.get(curr_pos, float('inf')):
                continue

            for action, (dx, dy) in directions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)

                if grid_size is not None:
                    w, h = grid_size
                    if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
                        continue

                if next_pos in walls_set:
                    continue

                step_cost = 1
                if step_costs and action in step_costs:
                    step_cost = step_costs[action]

                new_cost = cost + step_cost

                if new_cost < reached.get(next_pos, float('inf')):
                    reached[next_pos] = new_cost
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, next_pos, path + [action]))

        return None

    def astar_search(self, start_pos, goal_pos, walls, grid_size=None, heuristic_type='manhattan'):
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        walls_set = {tuple(w) for w in walls}

        if start == goal:
            return []

        pq = []
        reached_states = set()

        g_cost = 0
        if heuristic_type == 'manhattan':
            h_cost = self.manhattan_distance(start, goal)
        else:
            h_cost = self.euclidean_distance(start, goal)
        f_cost = g_cost + h_cost
        
        heapq.heappush(pq, (f_cost, g_cost, start, []))

        directions = [('Up', (0, 1)), ('Down', (0, -1)), ('Left', (-1, 0)), ('Right', (1, 0))]

        while pq:
            f_cost, g_cost, curr_pos, path = heapq.heappop(pq)

            if curr_pos == goal:
                return path

            if curr_pos in reached_states:
                continue

            reached_states.add(curr_pos)

            for action, (dx, dy) in directions:
                next_pos = (curr_pos[0] + dx, curr_pos[1] + dy)

                if grid_size is not None:
                    w, h = grid_size
                    if not (0 <= next_pos[0] < w and 0 <= next_pos[1] < h):
                        continue

                if next_pos in walls_set or next_pos in reached_states:
                    continue

                g_new = g_cost + 1
                if heuristic_type == 'manhattan':
                    h_new = self.manhattan_distance(next_pos, goal)
                else:
                    h_new = self.euclidean_distance(next_pos, goal)
                f_new = g_new + h_new
                
                heapq.heappush(pq, (f_new, g_new, next_pos, path + [action]))

        return None
