"""
    Problem 3 Template file
"""
import random
import math

import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for a problem setup given by the "RRT_dubins_problem" class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file "rrt_planning.py". Your implementation
   can be tested by running "RRT_dubins_problem.py" (see the "main()" function).
2. Read all class and function documentation in "RRT_dubins_problem.py" carefully.
   There are plenty of helper functions in the class that you should use.
3. Your solution must meet all the conditions specificed below.
4. Below are some DOs and DONTs for this problem.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random points
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out related issues and will be generously set.
2. The planning function must return a list of nodes that represent a collision free path
   from the start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must be a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation of the node to understand the terminology)
3. The returned path should be a valid list of nodes with a Dubins-style path connecting the nodes. 
   i.e. the list should have the start node at index 0 and goal node at index -1. 
   For all other indices i in the list, the parent node for node i should be at index i-1,  
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   "RRT_dubins_problem.map_area"

DO(s) and DONT(s)
-------------------
1. DO rename the file to rrt_planning.py for submission.
2. Do NOT change change the "planning" function signature.
3. Do NOT import anything other than what is already imported in this file.
4. We encourage you to write helper functions in this file in order to reduce code repetition
   but these functions can only be used inside the "planning" function.
   (since only the planning function will be imported)
"""
def cartesian_distance(node1, node2):
   return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def planning(rrt_dubins, display_map=False):
   # Fix Random Number Generator seed
   random.seed(1)

   # LOOP for max iterations
   for i in range(rrt_dubins.max_iter):

      # ON OCCASION, YOU RANDOMLY SET THE RANDOM STATE TO GOAL STATE SO THAT WE CAN CHECK IF THERE IS A PATH TO GOAL STATE
      if np.random.rand(1) > 0.2:
         rand_vehicle_state = rrt_dubins.Node(
            random.uniform(rrt_dubins.x_lim[0], rrt_dubins.x_lim[1]),
            random.uniform(rrt_dubins.y_lim[0], rrt_dubins.y_lim[1]),
            random.uniform(-math.pi, math.pi)
         )
      else:
         rand_vehicle_state = rrt_dubins.Node(rrt_dubins.goal.x, rrt_dubins.goal.y, rrt_dubins.goal.yaw)


      # Find an existing node nearest to the random vehicle state
      nearest_node = min(rrt_dubins.node_list, key=lambda node: cartesian_distance(node, rand_vehicle_state))

      # Propagate from nearest node to the random node
      new_node = rrt_dubins.propogate(nearest_node, rand_vehicle_state)


      # Check if the path between nearest node and random state has obstacle collision
      # Add the node to nodes_list if it is valid
      if rrt_dubins.check_collision(new_node): # true for safe, False if collision occurs
         new_node.parent = rrt_dubins.node_list[-1]
         rrt_dubins.node_list.append(new_node)  # Storing all valid nodes
         # maybe need node.parent just in case

      # Draw current view of the map
      # PRESS ESCAPE TO EXIT
      if display_map:
         rrt_dubins.draw_graph()

      # Check if new_node is close to goal -- check the last node in the node list (valid nodes only)
      if rrt_dubins.calc_dist_to_goal(rrt_dubins.node_list[-1].x, rrt_dubins.node_list[-1].y) < 0.1 and (abs(rrt_dubins.node_list[-1].yaw - rrt_dubins.goal.yaw) < 0.1/np.pi):
         print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
         return rrt_dubins.node_list

   if i == rrt_dubins.max_iter - 1:
      print('reached max iterations')

   return rrt_dubins.node_list # returns list so far


   # # Extract path from the node_list
   # path_node_list = [rrt_dubins.goal]
   # last_node = new_node
   # while last_node.parent is not None:
   #    path_node_list.append(last_node)
   #    last_node = last_node.parent
   # path_node_list.append(rrt_dubins.start)

   # return path_node_list[::-1]  # Return the reversed path


