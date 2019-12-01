import gym
import numpy as np
import math
from container_model import container
import gym.spaces as spaces
from matplotlib import pyplot as plt
import random

START_OWNSHIP = [1, 0, 0, 0, 0, 0, 0, 0, 0, 70]
GOAL_POSITION = [20_000, 0]
STEPS = 1000
h = 0.1
Lpp = 175

actions = [-10,-9,-8,-7,-6,-5,-4,-3,-2, -1,0,1,2,3,4,5,6,7,8,9,10]
enemy_xpos_array = []
enemy_ypos_array = []

# Constants
deg2rad = math.pi/180
rad2deg = 180/math.pi
m2km = 1/1000

MAX_YE = 500
MAX_INIT_YE = 100
DELTA = 10

def rot_matrix(alpha):
    return [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]

def transpose2D(matrix):
    temp = matrix[0][1]
    matrix[0][1] = matrix[1][0]
    matrix[1][0] = temp

    return matrix

def map_to_negpi_pi(angle):
    angle = angle%(2*math.pi)
    if angle > math.pi:
        angle = angle - 2*math.pi
    if angle < -math.pi:
        angle = angle + 2*math.pi

    return angle


class ContainerEnv(gym.Env):
    metadata = {'render.modes': ['container_vessel']}

    def __init__(self):

        self._seed = 99

        # Controller gains
        self.Kp = 1
        self.Kd = 10
        self.Ki = 0.001

        # Input to agent
        self.container_state =  []

        # For plotting
        self.xpos_array = []
        self.ypos_array = []
        self.time_array = [0]

        # For reward calculation: cross-track error and cross-track error derivative
        self.y_e = 0
        self.y_ed = 0


    def step(self, action):

        done = self._take_action(action)
        obs = self._get_obs()
        reward = self._get_reward()

        return obs, reward, done, {}

    def _get_obs(self):

        alpha_p = math.atan2(0 - GOAL_POSITION[0], 0 - GOAL_POSITION[1])
        psi_tilde = map_to_negpi_pi(self.container_state[5]-alpha_p)

        return [self.y_e/MAX_YE, self.y_ed, psi_tilde/math.pi, self.container_state[2], self.container_state[0], self.container_state[1]]


    def reset(self):

        ypos_init = random.randint(-MAX_INIT_YE, MAX_INIT_YE)
        psi_init = random.uniform(-math.pi, math.pi)

        self.container_state = [math.cos(psi_init), math.sin(psi_init), 0, 0, ypos_init, psi_init, 0, 0, 0, 60]
        self.xpos_array = [0]
        self.ypos_array = [ypos_init]
        self.y_e = self._dist_to_path()
        self.y_ed = 0

        return self._get_obs()


    def init_evaluate(self, psi, ypos):

        self.container_state = [math.cos(psi), math.sin(psi), 0, 0, ypos, psi, 0, 0, 0, 60]
        self.xpos_array = [0]
        self.ypos_array = [ypos]
        self.y_e = self._dist_to_path()
        self.y_ed = 0

        return self._get_obs()

    def render(self, mode='container_vessel', close=False):

        self._plot_position()

        return

    def _take_action(self, action):

        time = self.time_array[-1]

        y_e_prev = self.y_e

        n_c = 60
        xdot, _ = container(self.container_state, [actions[action]*deg2rad, n_c])

        for i in range(len(self.container_state)):
            self.container_state[i] = self.container_state[i] + h*xdot[i]
        self.container_state[5] = map_to_negpi_pi(self.container_state[5])

        time += h
        self.time_array.extend([time])

        self.y_e = self._dist_to_path()

        # Calculate change in
        self.y_ed = (self.y_e - y_e_prev)/h
        self.xpos_array.extend([self.container_state[3]])
        self.ypos_array.extend([self.container_state[4]])


        if abs(self.container_state[4]) > MAX_YE:
            return True

        else:
            return False


    def _dist_to_path(self):
        alpha = math.atan2(0 - GOAL_POSITION[0], 0 - GOAL_POSITION[1])

        rot = rot_matrix(alpha)
        rot_T = transpose2D(rot)

        return rot_T[0][0]*self.container_state[3] + rot_T[0][1]*self.container_state[4]


    def _get_reward(self):

        alpha = math.atan2(0 - GOAL_POSITION[0], 0 - GOAL_POSITION[1])
        psi = self.container_state[5]

        #if self.y_e > 50:
        #    return -1

        if abs(map_to_negpi_pi(psi-alpha)) < math.pi/4 and self.y_e < 50:
            #std = 30
            #amp = 1

            #reward = amp * math.e**(-(self.y_e**2)/(2*std**2))

            reward = 1-(1/50)*self.y_e
            reward = (1-self.y_ed/8)*reward

            return reward

        else:
            return 0


    def _plot_position(self):

        params = {'backend': 'ps',
          'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True}
        plt.rcParams.update(params)

        plt.plot([x*m2km for x in self.xpos_array], [y*m2km for y in self.ypos_array])
        plt.plot([0, GOAL_POSITION[0]*m2km], [0, GOAL_POSITION[1]*m2km], linestyle='--')

        plt.title("Position of ship",
                  fontsize=12,fontweight="bold")
        plt.xlabel("X position [km]",fontsize=9,fontweight="bold")
        plt.ylabel("Y position [km]",fontsize=9,fontweight="bold")

        plt.show()
