# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()#初始化栈，存储（状态，路径）元组
    stack.push((problem.getStartState(),[]))#起始状态入栈，初始路径为空
    visited = set()
    
    while not stack.isEmpty():
      #弹出栈顶结点
      current_state,path = stack.pop()
      
      #检查是否为目标状态
      if problem.isGoalState(current_state):
        return path
      
      #如果未访问过该状态
      if current_state not in visited:
        #标记为已访问
        visited.add(current_state)
        
        #扩展所有后继结点
        for successor, action, _ in problem.getSuccessors(current_state):
          if successor not in visited:
            #新路径 = 原路径 + 当前动作
            new_path = path + [action]
            stack.push((successor,new_path))
    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #1、初始化队列，存储当前状态以及到该状态的动态路径
    queue = util.Queue()
    queue.push((problem.getStartState(),[]))
    
    #2、初始化已访问集合，避免重复探索
    visited = set()
    
    while not queue.isEmpty():
      #3、取出队首节点，先入队的先处理
      current_state,path = queue.pop()
      
      #4、检查是否到达目标状态，若是则返回路径
      if problem.isGoalState(current_state):
        return path
        
      #5、若当前状态未访问过，才扩展其“后继结点”
      if current_state not in visited:
        visited.add(current_state)
      #遍历所有后继状态
        for next_state, action, _ in problem.getSuccessors(current_state):
          if next_state not in visited:
            new_path = path + [action]#拼接新动作
            queue.push((next_state,new_path))
    
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #初始化优先队列（按总代价升序），存储（总代价，当前状态，动作路径）
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((0,start_state,[]),0)#优先级为总代价
    
    #记录每个状态的最小总代价（避免重复扩展高代价路径）
    cost_so_far = {start_state:0}
    
    while not priority_queue.isEmpty():
      #取出总代价最小的节点（UCS核心：优先扩展代价最低的节点）
      current_cost,current_state,path = priority_queue.pop()
      
      #检查是否为目标状态
      if problem.isGoalState(current_state):
        return path
      #遍历所有后续节点
      for successor,action,step_cost in problem.getSuccessors(current_state):
        new_cost = current_cost + step_cost
        
        if successor not in cost_so_far or new_cost < cost_so_far[successor]:
          cost_so_far[successor] = new_cost
          new_path = path + [action]
          priority_queue.push((new_cost,successor,new_path),new_cost)
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #初始化优先队列，存储（当前g值，当前状态，路径），优先级为f = g + h
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    #初始状态，g=0，路径为空，优先级f = 0 + 启发函数估计值
    priority_queue.push((0,start_state,[]),0 + heuristic(start_state,problem))
    
    #记录每个状态的最小g值，避免重复扩展高代价路径
    cost_so_far = {start_state:0}
    
    while not priority_queue.isEmpty():
      #弹出f值最小的节点（优先级最高）
      current_g,current_state,path = priority_queue.pop()
      
      #若当前状态是目标状态，返回路径：
      if problem.isGoalState(current_state):
        return path
      
      #扩展所有后继结点
      for successor,action,step_cost in problem.getSuccessors(current_state):
        new_g = current_g + step_cost
        
        if successor not in cost_so_far or new_g < cost_so_far[successor]:
          cost_so_far[successor] = new_g
          new_path = path + [action]
          f_value = new_g + heuristic(successor,problem)
          
          priority_queue.push((new_g,successor,new_path),f_value)
          
    return []
        
      
    
    
    #记录每个状态的最小g值，
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
