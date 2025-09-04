"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random

#A unique identifier used to track nodes; each newly created MCTS_Node receives a unique ID.
node_cnt = 0

#Debug output, see each step's selection
def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class MCTS_Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self) -> None:
        super().__init__()

        global node_cnt
        self.id = node_cnt
        node_cnt += 1

        self.rollout_id = None

    #Used to mark the current rollout for easy differentiation between different rollouts.
    def set_rollout_id(self, rollout_id: int):
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int):
        "All possible successors of this board state"
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        raise NotImplementedError

    @abstractmethod
    def skip_backprop(self):
        "If True, the reward of this node will not be accumulated in the backpropagation step."
        raise NotImplementedError


class MCTS_Searcher:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        discount: float,
        verbose: bool = False,
    ):
        #Q：Record the cumulative rewards for each node.
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)
        #N：Record the number of visits to each node.
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)
        #parent2children：Record the child nodes of each node.
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = dict()  # children of each node

        #! explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        #explored_nodes：recording nodes that have already been explored.
        self.explored_nodes = set()
        #exploration_weight：Exploration weight, controlling the balance between exploration and exploitation.
        self.exploration_weight = exploration_weight
        #weight_scheduler：Strategies for Controlling Changes in Exploration Weighting.
        self.weight_scheduler = weight_scheduler
        #num_rollouts：Simulated number of rounds.
        self.num_rollouts = num_rollouts
        #discount：Discount factor, used for reward decay.
        self.discount = discount
        #verbose：Enable debug output. You can see each step's selection.
        self.verbose = verbose

        # A unique identifier used to track nodes; each newly created MCTS_Node is assigned a unique ID.
        global node_cnt
        node_cnt = 0

    #Execute one round of Monte Carlo tree search
    def do_rollout(self, root_node: MCTS_Node, rollout_id: int):
        "Make the tree one layer better. (Train for one iteration.)"
        verbose_print("==> Selecting a node...", self.verbose)
        path_1 = self._select(root_node, rollout_id)
        leaf = path_1[-1]
        verbose_print(f"==> Expanding node {leaf.id}...", self.verbose)
        self._expand(leaf, rollout_id)
        verbose_print(f"==> Simulating node {leaf.id}...", self.verbose)
        path_2 = self._simulate(leaf, rollout_id)
        verbose_print(f"==> Backpropagating...", self.verbose)
        self._backpropagate(path_1 + path_2)
        return leaf.id

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.parent2children.keys():
                return path

            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                return path

            node = self._uct_select(node, rollout_id)

    def _expand(self, node: MCTS_Node, rollout_id: int):
        if node in self.explored_nodes:
            return  # already expanded

        if node.is_terminal():
            self.explored_nodes.add(node)
            return  # terminal node is non-expandable
        self.parent2children[node] = node.find_children(rollout_id)


    def _simulate(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Returns the reward for a random simulation (to completion) of `node`"
        path = []
        cur_node = node
        while True:
            if cur_node.is_terminal():
                self.explored_nodes.add(node)
                return path

            if cur_node not in self.parent2children.keys():
                self.parent2children[cur_node] = cur_node.find_children(rollout_id)

            cur_node = random.choice(self.parent2children[cur_node])  # randomly select a child
            path.append(cur_node)

    def _backpropagate(self, path: List[MCTS_Node]):
        "Send the reward back up to the ancestors of the leaf"
        leaf = path[-1]
        reward = leaf.calculate_reward()
        verbose_print(f"reward: {reward}", self.verbose)
        for node in reversed(path):
            self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "const":
            return self.exploration_weight

    def _uct_select(self, node: MCTS_Node, rollout_id: int):
        "Select a child of node, balancing exploration & exploitation"

        # All children of the node should already be expanded
        assert all(n in self.explored_nodes for n in self.parent2children[node])

        return max(
            self.parent2children[node], key=lambda n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
        )

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
                return 999
            else:
                weight = self._get_weight(rollout_id)
                return self.Q[node] / self.N[node] + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])

    def verbose_draw_tree(self, root_node: MCTS_Node, verbose: bool, selected_node: int = 0):
        """Draw the tree structure starting from the root node with visual formatting."""
        if not verbose:
            return
        
        # ANSI escape codes for colors
        COLOR_RED = '\033[91m'
        COLOR_RESET = '\033[0m'
        
        # Print root node with special handling, include its accumulated reward from self.Q
        root_reward = self.Q.get(root_node, 0.0)  # Retrieve the reward from self.Q
        root_info = f"id: {root_node.id}, node_type: {root_node.node_type}, reward: {root_reward}"
        if root_node.id == selected_node:
            root_info = f"{COLOR_RED}{root_info}{COLOR_RESET}"
        print(root_info)
        
        def _print_node(node, prefix: str, is_last: bool):
            """Recursive helper to print tree nodes with proper formatting."""
            # Retrieve node's accumulated reward from self.Q
            node_reward = self.Q.get(node, 0.0)
            node_info = f"id: {node.id}, node_type: {node.node_type}, reward: {node_reward}"
            if node.id == selected_node:
                node_info = f"{COLOR_RED}{node_info}{COLOR_RESET}"
            
            # Build connection symbols
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{node_info}")
            
            # Prepare new prefix for children
            extension = "    " if is_last else "│   "
            new_prefix = prefix + extension
            
            # Recursively print children
            children = getattr(node, 'children', [])
            for i, child in enumerate(children):
                _print_node(child, new_prefix, i == len(children)-1)
        
        # Start recursive printing from root's children
        children = getattr(root_node, 'children', [])
        for i, child in enumerate(children):
            _print_node(child, "", i == len(children)-1)
    

    def select_solution(self, root_node: MCTS_Node):
        current_node = root_node
        path = [current_node]  

        while hasattr(current_node, 'children') and current_node.children:
            children = current_node.children
            if not children:
                break

            best_child = None
            max_q = -float('inf')
            for child in children:
                q = self.Q.get(child, 0.0) 
                if q > max_q or (q == max_q and best_child is None):
                    max_q = q
                    best_child = child

            if best_child:
                current_node = best_child
                path.append(current_node)
            else:
                break

        best_leaf = current_node

        return best_leaf

    def select_final_solution(self, root_node: MCTS_Node, verbose: bool):
        current_node = root_node
        path = [current_node] 

        while hasattr(current_node, 'children') and current_node.children:
            children = current_node.children
            if not children:
                break

            best_child = None
            max_q = -float('inf')
            for child in children:
                q = self.Q.get(child, 0.0)  
                if q > max_q or (q == max_q and best_child is None):
                    max_q = q
                    best_child = child

            if best_child:
                current_node = best_child
                path.append(current_node)
            else:
                break 

        best_leaf = current_node

        if verbose:
            print("\nUse Q select path:")
            for node in path:
                q = self.Q.get(node, 0.0)
                n = self.N.get(node, 0)
                ratio = q / n if n != 0 else 0.0
                print(f"Node ID: {node.id}, Type: {node.node_type}, Q: {q:.2f}, N: {n}, Q/N: {ratio:.2f}")
            print(f"Best solution: {getattr(best_leaf, 'answer', 'No answer found.')}")

        return getattr(best_leaf, 'answer', None)