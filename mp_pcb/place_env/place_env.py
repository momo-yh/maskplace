import math
import numpy as np

import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import sys
sys.path.append("..")
from comp_res import comp_res 


class PlaceEnv(gym.Env):

    def __init__(self, placedb, grid = 224):#TODO:grid_size 
        
        self.placedb = placedb
        # need to get GCN vector and CNN
        print("grid*grid", grid*grid)
        print("placedb.node_cnt", self.placedb.node_cnt)
        print("placedb.net_cnt", self.placedb.net_cnt)
        assert grid*grid >= self.placedb.node_cnt 
        self.grid = grid
        self.grid = grid

        self.placed_num_macro = self.placedb.node_cnt #TODO: this name not good
        self.num_net = self.placedb.net_cnt
        self.node_name_list = self.placedb.node_id_to_name

        self.action_space = spaces.Discrete(grid*grid)
        self.state = None
        self.net_min_max_ord = {}
        self.node_pos = {}
        self.net_placed_set = {}
        self.last_reward = 0
        self.num_macro_placed = 0

        ##TODO: self.ratio not considered
        self.ratio = 0.1
        print("self.ratio = {:.2f}".format(self.ratio))
    
    def reset(self):
        self.num_macro_placed = 0
        canvas = np.zeros((self.grid, self.grid))
        self.node_pos = {}

        '''[4] min_x, max_x, min_y, max_y'''
        self.net_min_max_ord = {}
        self.net_fea = np.zeros((self.num_net, 4)) 
        self.net_fea[:, 0] = 0
        self.net_fea[:, 1] = 1.0
        self.net_fea[:, 2] = 0
        self.net_fea[:, 3] = 1.0
        # self.rudy = np.zeros((self.grid, self.grid))
        ##TODO:port not considered
        # for port_name in self.placedb.port_to_net_dict:
        #     for net_name in self.placedb.port_to_net_dict[port_name]:
        #         pin_x = round(self.placedb.port_info[port_name]['width'] / self.ratio)
        #         pin_y = round(self.placedb.port_info[port_name]['height'] / self.ratio)
        #         if net_name in self.net_min_max_ord:
        #             '''min max x y of net'''
        #             if pin_x > self.net_min_max_ord[net_name]['max_x']:
        #                 self.net_min_max_ord[net_name]['max_x'] = pin_x
        #                 self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
        #             elif pin_x < self.net_min_max_ord[net_name]['min_x']:
        #                 self.net_min_max_ord[net_name]['max_y'] = pin_y
        #                 self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
        #             if pin_y > self.net_min_max_ord[net_name]['max_y']:
        #                 self.net_min_max_ord[net_name]['max_y'] = pin_y
        #                 self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
        #             elif pin_y < self.net_min_max_ord[net_name]['min_y']:
        #                 self.net_min_max_ord[net_name]['min_y'] = pin_y
        #                 self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
        #         else:
        #             self.net_min_max_ord[net_name] = {}
        #             self.net_min_max_ord[net_name]['max_x'] = pin_x
        #             self.net_min_max_ord[net_name]['min_x'] = pin_x
        #             self.net_min_max_ord[net_name]['max_y'] = pin_y
        #             self.net_min_max_ord[net_name]['min_y'] = pin_y
        #             self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
        #             self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
        #             self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
        #             self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
        
        '''reset wire mask'''
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))

        '''position mask the area that is not allowed to place, 
        The occupied area is then marked as 1 in the mask array.'''
        next_x = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['width'] / self.ratio))
        next_y = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed]]['height'] / self.ratio))
        mask = self.get_mask(next_x, next_y)
        next_x_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['width'] / self.ratio))
        next_y_2 = math.ceil(max(1, self.placedb.node_info[self.node_name_list[self.num_macro_placed+1]]['height'] / self.ratio))
        mask_2 = self.get_mask(next_x_2, next_y_2)

        '''reset net placed '''
        self.net_placed_set = {}
        for net_name in self.placedb.net_info:
            self.net_placed_set[net_name] = set()
        
        '''state representation'''##TODO: state representation can be optimized
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(), 
            net_img.flatten(), mask.flatten(), net_img_2.flatten(), mask_2.flatten(), 
            np.array([next_x/self.grid, next_y/self.grid])), axis = 0)

        return self.state
    
#TODO: exact save_fig and get mask function to helper.py
    def save_fig(self, file_path):
        '''save the placement result as a figure'''
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            ax1.add_patch(
                patches.Rectangle(
                    (x/self.grid, y/self.grid),   # (x,y)
                    size_x/self.grid,          # width
                    size_y/self.grid, linewidth=1, edgecolor='k',
                )
            )
        fig1.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close()
    def save_wiremask(self, file_path):
        # Plotting the net_img array
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # Plotting the net_img array
        ax.imshow(self.get_net_img())
        fig.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close()
    # PositionMask
    def get_mask(self, next_x, next_y):
        mask = np.zeros((self.grid, self.grid))
        for node_name in self.node_pos:
            startx = max(0, self.node_pos[node_name][0] - next_x + 1)
            starty = max(0, self.node_pos[node_name][1] - next_y + 1)
            endx = min(self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1, self.grid - 1)
            mask[startx: endx + 1, starty : endy + 1] = 1
        
        mask[self.grid - next_x + 1:,:] = 1 
        mask[:, self.grid - next_y + 1:] = 1
        '''exclude the area that is not allowed to place, The occupied area is then marked as 1 in the mask array.'''
        return mask
    # WireMask
    def get_net_img(self, is_next_next = False):
        net_img = np.zeros((self.grid, self.grid))
        if not is_next_next:
            next_node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        elif self.num_macro_placed + 1 < len(self.placedb.node_id_to_name):
            next_node_name = self.placedb.node_id_to_name[self.num_macro_placed + 1]
        else:
            return net_img

        for net_name in self.placedb.node_to_net_dict[next_node_name]:
            if net_name in self.net_min_max_ord:
                # HPWL with pin offset
                delta_pin_x = round((self.placedb.node_info[next_node_name]['width']/2 + \
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["x_offset"])/self.ratio)
                delta_pin_y = round((self.placedb.node_info[next_node_name]['height']/2 + \
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["y_offset"])/self.ratio)
                start_x = self.net_min_max_ord[net_name]['min_x'] - delta_pin_x
                end_x = self.net_min_max_ord[net_name]['max_x'] - delta_pin_x
                start_y = self.net_min_max_ord[net_name]['min_y'] - delta_pin_y
                end_y = self.net_min_max_ord[net_name]['max_y'] - delta_pin_y
                start_x = min(start_x, self.grid)
                start_y = min(start_y, self.grid)
                if not 'weight' in self.placedb.net_info[net_name]:
                    weight = 1.0
                else:
                    weight = self.placedb.net_info[net_name]['weight']
                for i in range(0, start_x):
                    net_img[i, :] += (start_x - i) * weight
                for i in range(end_x+1, self.grid):
                    net_img[i, :] +=  (i- end_x) * weight
                for j in range(0, start_y):
                    net_img[:, j] += (start_y - j) * weight
                for j in range(end_y+1, self.grid):
                    net_img[:, j] += (j - start_y) * weight
        return net_img

    def step(self, action):
        '''chech if the action is valid'''
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        '''extract the state'''
        canvas = self.state[1: 1+self.grid*self.grid].reshape(self.grid, self.grid)
        mask = self.state[1+self.grid*self.grid*2: 1+self.grid*self.grid*3].reshape(self.grid, self.grid)
        
        '''get the action'''
        reward = 0
        x = round(action // self.grid)
        y = round(action % self.grid)
    
        if mask[x][y] == 1: #placing in 1 is overlapping
            reward += -200000

        '''get the size of the next macro to place'''        
        node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        size_x = math.ceil(max(1, self.placedb.node_info[node_name]['width']/self.ratio))
        size_y = math.ceil(max(1, self.placedb.node_info[node_name]['height']/self.ratio))

        ## TODO: may be not needed?
        # assert abs(size_x - self.state[-2]*self.grid) < 1e-5
        # assert abs(size_y - self.state[-1]*self.grid) < 1e-5

        '''place in canvas for state representation'''
        ##TODO: seemly wrong here
        canvas[x : x+size_x, y : y+size_y] = 1.0
        canvas[x : x + size_x, y] = 0.5
        if y + size_y -1 < self.grid:
            canvas[x : x + size_x, max(0, y + size_y -1)] = 0.5
        canvas[x, y: y + size_y] = 0.5
        if x + size_x - 1 < self.grid:
            canvas[max(0, x+size_x-1), y: y + size_y] = 0.5
        self.node_pos[self.node_name_list[self.num_macro_placed]] = (x, y, size_x, size_y)

        '''reward for wirelength'''
        # for net_name in self.placedb.node_to_net_dict[node_name]:
        #     self.net_placed_set[net_name].add(node_name)
        #     pin_x = round((x * self.ratio + self.placedb.node_info[node_name]['width']/2 + \
        #             self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"])/self.ratio)
        #     pin_y = round((y * self.ratio + self.placedb.node_info[node_name]['height']/2 + \
        #         self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"])/self.ratio)
        #     if net_name in self.net_min_max_ord:
        #         start_x = self.net_min_max_ord[net_name]['min_x']
        #         end_x = self.net_min_max_ord[net_name]['max_x']
        #         start_y = self.net_min_max_ord[net_name]['min_y']
        #         end_y = self.net_min_max_ord[net_name]['max_y']
        #         delta_x = end_x - start_x
        #         delta_y = end_y - start_y
        #         if delta_x > 0 or delta_y > 0:
        #             self.rudy[start_x : end_x +1, start_y: end_y +1] -= 1/(delta_x+1) + 1/(delta_y+1) 
        #         weight = 1.0
        #         if 'weight' in self.placedb.net_info[net_name]:
        #             weight = self.placedb.net_info[net_name]['weight']
 
        #         if pin_x > self.net_min_max_ord[net_name]['max_x']:
        #             reward += weight * (self.net_min_max_ord[net_name]['max_x'] - pin_x)
        #             self.net_min_max_ord[net_name]['max_x'] = pin_x
        #             self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
        #         elif pin_x < self.net_min_max_ord[net_name]['min_x']:
        #             reward += weight * (pin_x - self.net_min_max_ord[net_name]['min_x'])
        #             self.net_min_max_ord[net_name]['min_x'] = pin_x
        #             self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
        #         if pin_y > self.net_min_max_ord[net_name]['max_y']:
        #             reward += weight * (self.net_min_max_ord[net_name]['max_y'] - pin_y)
        #             self.net_min_max_ord[net_name]['max_y'] = pin_y
        #             self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
        #         elif pin_y < self.net_min_max_ord[net_name]['min_y']:
        #             reward += weight * (pin_y - self.net_min_max_ord[net_name]['min_y'])
        #             self.net_min_max_ord[net_name]['min_y'] = pin_y
        #             self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
        #         start_x = self.net_min_max_ord[net_name]['min_x']
        #         end_x = self.net_min_max_ord[net_name]['max_x']
        #         start_y = self.net_min_max_ord[net_name]['min_y']
        #         end_y = self.net_min_max_ord[net_name]['max_y']
        #         delta_x = end_x - start_x
        #         delta_y = end_y - start_y
        #         self.rudy[start_x : end_x +1, start_y: end_y +1] += 1/(delta_x+1) + 1/(delta_y+1) 
        #     else:
        #         self.net_min_max_ord[net_name] = {}
        #         self.net_min_max_ord[net_name]['max_x'] = pin_x
        #         self.net_min_max_ord[net_name]['min_x'] = pin_x
        #         self.net_min_max_ord[net_name]['max_y'] = pin_y
        #         self.net_min_max_ord[net_name]['min_y'] = pin_y
        #         start_x = self.net_min_max_ord[net_name]['min_x']
        #         end_x = self.net_min_max_ord[net_name]['max_x']
        #         start_y = self.net_min_max_ord[net_name]['min_y']
        #         end_y = self.net_min_max_ord[net_name]['max_y']
        #         self.net_fea[self.placedb.net_info[net_name]['id']][1] = pin_x / self.grid
        #         self.net_fea[self.placedb.net_info[net_name]['id']][0] = pin_x / self.grid
        #         self.net_fea[self.placedb.net_info[net_name]['id']][3] = pin_y / self.grid
        #         self.net_fea[self.placedb.net_info[net_name]['id']][2] = pin_y / self.grid
        #         reward += 0
        '''reward of HPWl for placed net'''#TODO: reward for net not placed
        # reward = comp_res(self.placedb, node_pos, 1.0)
        reward += 0

        '''get wire mask'''
        self.num_macro_placed += 1
        net_img = np.zeros((self.grid, self.grid))
        net_img_2 = np.zeros((self.grid, self.grid))
        ##TODO: may be not needed? seemly work for locked-node
        # if self.num_macro_placed < self.placed_num_macro: 
        #     net_img = self.get_net_img()
        #     net_img_2 = self.get_net_img(is_next_next= True)
        #     if net_img.max() >0 or net_img_2.max()>0:
        #         net_img /= max(net_img.max(), net_img_2.max())
        #         net_img_2 /= max(net_img.max(), net_img_2.max())

        '''done signal update'''
        if self.num_macro_placed == self.placed_num_macro:
            done = True
        else:
            done = False

        '''get position mask'''
        mask = np.ones((self.grid, self.grid))
        mask_2 = np.ones((self.grid, self.grid))
        if not done: # get next macro size and pre-mask the solution
            next_x = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['width']/self.ratio))
            next_y = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['height']/self.ratio))
            mask = self.get_mask(next_x, next_y)
            if self.num_macro_placed + 1 < self.placed_num_macro:
                next_x_2 = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed+1]]['width']/self.ratio))
                next_y_2 = math.ceil(max(1, self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed+1]]['height']/self.ratio))
                mask_2 = self.get_mask(next_x_2, next_y_2)
        else:
            next_x = 0
            next_y = 0

        '''generate state representation'''
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(), 
            net_img.flatten(), mask.flatten(), net_img_2.flatten(), mask_2.flatten(),
            np.array([next_x/self.grid, next_y/self.grid])), axis = 0)
        
        return self.state, reward, done, {"raw_reward": reward, "net_img": net_img, "mask": mask}
    
    