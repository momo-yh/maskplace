import os
from itertools import combinations
from operator import itemgetter

from kiutils.board import Board
from kiutils.items import fpitems

def read_board(board):
    '''read board_info from board object'''
    node_info = {}
    node_cnt = 0

    net_info = {}
    net_name = None

    port_info = {}
    port_name = None

    boundary = None

    ## read board boundary
    gr_x = []
    gr_y = []
    gr_boundary_range = 0.06
    for gr_line in board.graphicItems:
        gr_x.append(gr_line.end.X)
        gr_y.append(gr_line.end.Y)
        gr_x.append(gr_line.start.X)
        gr_y.append(gr_line.start.Y)
    boundary = [max(gr_x)-gr_boundary_range, min(gr_x)+gr_boundary_range,
                max(gr_y)-gr_boundary_range, min(gr_y)+gr_boundary_range]

    ## read nets info
    for net in board.nets:
        net_name = net.name
        net_info[net_name] = {}
        net_info[net_name]['id'] = net.number
        net_info[net_name]["nodes"] = {}
        net_info[net_name]["ports"] = {}

    ## read footprint info
    for footprint in board.footprints:
        angle = footprint.position.angle
        raw_x = footprint.position.X
        raw_y = footprint.position.Y
        x = raw_x - boundary[1]
        y = raw_y - boundary[3]

        all_x = []
        all_y = []
        for items in footprint.graphicItems:
            if type(items) == fpitems.FpPoly:
                for point in items.coordinates:
                    all_x.append(point.X)
                    all_y.append(point.Y)
                wid = items.stroke.width
            if type(items) == fpitems.FpText:
                if items.type == 'reference':
                    footprint_name = items.text
        width = max(all_x) - min(all_x) + wid + 0.2
        height = max(all_y) - min(all_y) + wid + 0.2
        node_name = footprint_name
        node_info[node_name] = {"id": node_cnt, "x": x , "y": y, 
                                "width": width, "height": height,
                                "raw_x": x, "raw_y": y,
                                "angle": angle}
        node_cnt += 1

        ## read pad info to link nodes to net_info
        for pad in footprint.pads:
            pad_num = pad.number
            x_offset = pad.position.X
            y_offset = pad.position.Y
            if pad.net:
                net_name = pad.net.name
                if node_name not in net_info[net_name]["nodes"]:
                    net_info[net_name]["nodes"][node_name] = {}
                if "pads" not in net_info[net_name]["nodes"][node_name]:
                    net_info[net_name]["nodes"][node_name]["pads"] = {}
                net_info[net_name]["nodes"][node_name]["pads"][pad_num] = {}
                net_info[net_name]["nodes"][node_name]["pads"][pad_num]["x_offset"] = x_offset 
                net_info[net_name]["nodes"][node_name]["pads"][pad_num]["y_offset"] = y_offset   
    

    return node_info, net_info, port_info, boundary

def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_node_id_to_name(node_info, node_to_net_dict):
    node_name_and_num = []
    for node_name in node_info:
        node_name_and_num.append((node_name, len(node_to_net_dict[node_name])))
    node_name_and_num = sorted(node_name_and_num, key=itemgetter(1), reverse = True)
    node_id_to_name = [node_name for node_name, _ in node_name_and_num]
    for i, node_name in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    return node_id_to_name

'''
TOOD: 'get_node_id_to_name_topology' considers the topology of the network. 
    It forms an adjacency graph based on network connections, 
    and then it iteratively selects nodes based on a set of criteria. 
    The criteria involve factors such as the number of 
    connections, node area, and a benchmark-specific formula.
'''
# def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    # node_id_to_name = []
    # adjacency = {}
    # for net_name in net_info:
    #     for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
    #         if node_name_1 not in adjacency:
    #             adjacency[node_name_1] = set()
    #         if node_name_2 not in adjacency:
    #             adjacency[node_name_2] = set()
    #         adjacency[node_name_1].add(node_name_2)
    #         adjacency[node_name_2].add(node_name_1)

    # visited_node = set()

    # node_net_num = {}
    # for node_name in node_info:
    #     node_net_num[node_name] = len(node_to_net_dict[node_name])
    
    # node_net_num_fea= {}
    # node_net_num_max = max(node_net_num.values())
    # print("node_net_num_max", node_net_num_max)
    # for node_name in node_info:
    #     node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    # node_area_fea = {}
    # node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    # node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    # print("node_area_max = {}".format(node_area_max))
    # for node_name in node_info:
    #     node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    # if "V" in node_info:
    #     add_node = "V"
    #     visited_node.add(add_node)
    #     node_id_to_name.append((add_node, node_net_num[add_node]))
    #     node_net_num.pop(add_node)
    
    # add_node = max(node_net_num, key = lambda v: node_net_num[v])
    # visited_node.add(add_node)
    # node_id_to_name.append((add_node, node_net_num[add_node]))
    # node_net_num.pop(add_node)

    # while len(node_id_to_name) < len(node_info):
    #     candidates = {}
    #     for node_name in visited_node:
    #         if node_name not in adjacency:
    #             continue
    #         for node_name_2 in adjacency[node_name]:
    #             if node_name_2 in visited_node:
    #                 continue
    #             if node_name_2 not in candidates:
    #                 candidates[node_name_2] = 0
    #             candidates[node_name_2] += 1
    #     for node_name in node_info:
    #         if node_name not in candidates and node_name not in visited_node:
    #             candidates[node_name] = 0
    #     if len(candidates) > 0:
    #         if benchmark == "bigblue3":
    #             add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
    #                 node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
    #         else:
    #             add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
    #                 node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)

    #     else:
    #         if benchmark == "bigblue3":
    #             add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
    #         else:
    #             add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

    #     visited_node.add(add_node)
    #     node_id_to_name.append((add_node, node_net_num[add_node])) 
    #     node_net_num.pop(add_node)
    # for i, (node_name, _) in enumerate(node_id_to_name):
    #     node_info[node_name]["id"] = i
    # node_id_to_name_res = [x for x, _ in node_id_to_name]
    # return node_id_to_name_res


class PlaceDB():
    def __init__(self, benchmark = "bm1"):

        bm_path = "PCBBenchmarks/{0}/{0}.unrouted.kicad_pcb".format(benchmark)
        self.board = Board().from_file(bm_path)

        #basic board info
        self.node_info, self.net_info, self.port_info, self.boundary = read_board(self.board)
        self.node_cnt = len(self.node_info)
        self.net_cnt = len(self.net_info)

        self.port_to_net_dict = {} #TODO
        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        self.node_id_to_name = get_node_id_to_name(self.node_info, self.node_to_net_dict)

        #TODO: get raw node_pos from board
        
        # board boundary
        self.max_height = self.boundary[2] - self.boundary[3]
        self.max_width = self.boundary[0] - self.boundary[1]

        #print info
        print("[PlaceDB_info]:using benchmark {}".format(benchmark))
        self.debug_str()



    def debug_str(self):
        print("[PlaceDB_info]:node_cnt = {}".format(len(self.node_info)))
        print("[PlaceDB_info]:net_cnt = {}".format(len(self.net_info)))
        print("[PlaceDB_info]:port_cnt = {}".format(len(self.port_info)))
        print("[PlaceDB_info]:board boundary:", self.boundary)
        print("[PlaceDB_info]:node_id_to_name = {}".format(self.node_id_to_name))



if __name__ == "__main__":
    placedb = PlaceDB('bm7')

