def comp_res(placedb, node_pos, ratio):
    '''Compute the HPWL and minimum spanning tree cost of the placement.
    Args:
        placedb: the placement database.
        node_pos: the position of the nodes.
        ratio: the ratio of the placement.'''

    hpwl = 0.0
    cost = 0.0
    for net_name in placedb.net_info:
        if len(placedb.net_info[net_name]["nodes"]) == 0:
            hpwl_tmp = 0
        else:
            x_list = []
            y_list = []
            for node_name in placedb.net_info[net_name]["nodes"]:
                if node_name not in node_pos:
                    raise AssertionError("node {} : position not given in node_pos".format(node_name))
                for pad in placedb.net_info[net_name]["nodes"][node_name]["pads"]:
                    pad_x_offset = placedb.net_info[net_name]["nodes"][node_name]["pads"][pad]["x_offset"]
                    pad_y_offset = placedb.net_info[net_name]["nodes"][node_name]["pads"][pad]["y_offset"]
                    if (placedb.node_info[node_name]["angle"] == 0):
                        x_list.append(node_pos[node_name][0] + pad_x_offset)
                        y_list.append(node_pos[node_name][1] + pad_y_offset)
                    elif (placedb.node_info[node_name]["angle"] == 90):
                        x_list.append(node_pos[node_name][0] + pad_y_offset)
                        y_list.append(node_pos[node_name][1] - pad_x_offset)
                    elif (placedb.node_info[node_name]["angle"] == 180):
                        x_list.append(node_pos[node_name][0] - pad_x_offset)
                        y_list.append(node_pos[node_name][1] - pad_y_offset)
                    elif (placedb.node_info[node_name]["angle"] == 270) or (placedb.node_info[node_name]["angle"] == -90):
                        x_list.append(node_pos[node_name][0] - pad_y_offset)
                        y_list.append(node_pos[node_name][1] + pad_x_offset)
                    else:## angle not given , default 0
                        x_list.append(node_pos[node_name][0] + pad_x_offset)
                        y_list.append(node_pos[node_name][1] + pad_y_offset)
            # #TODO port not considered
            # for port_name in placedb.net_info[net_name]["ports"]:
            #     for pad in placedb.net_info[net_name]["ports"][port_name]["pads"]:
            #         pad_x_offset = placedb.net_info[net_name]["ports"][port_name]["pads"][pad]["x_offset"]
            #         pad_y_offset = placedb.net_info[net_name]["ports"][port_name]["pads"][pad]["y_offset"]
            #         x_list.append(node_pos[port_name][0] + pad_x_offset)
            #         y_list.append(node_pos[port_name][1] + pad_y_offset)
            max_x = max(x_list)
            min_x = min(x_list)
            max_y = max(y_list)
            min_y = min(y_list)
            if min_x <= placedb.max_height:
                hpwl_tmp = (max_x - min_x) + (max_y - min_y)
            else:
                hpwl_tmp = 0
            if "weight" in placedb.net_info[net_name]:
                hpwl_tmp *= placedb.net_info[net_name]["weight"]
            hpwl += hpwl_tmp
        # #debug_info    
        # print("id:{} {}, hpwl_tmp:{}".format(placedb.net_info[net_name]["id"], net_name, hpwl_tmp))

        # ##TODO: prim cost is not accurate, what is needed?
        # from prim import prim_real
        # 
        # net_node_set = set.union(set(placedb.net_info[net_name]["nodes"]),
        #                     set(placedb.net_info[net_name]["ports"]))
        # ## delete useless nodes
        # for net_node in list(net_node_set):
        #     if net_node not in node_pos and net_node not in placedb.port_info:
        #         net_node_set.discard(net_node)

        # prim_cost = prim_real(net_node_set, node_pos, placedb.net_info[net_name]["nodes"], ratio, placedb.node_info, placedb.port_info)
        # if "weight" in placedb.net_info[net_name]:
        #     prim_cost *= placedb.net_info[net_name]["weight"]
        # # assert hpwl_tmp <= prim_cost +1e-5
        # cost += prim_cost
        
    return hpwl, cost

if __name__ == "__main__":
    import place_db
    placedb = place_db.PlaceDB("bm7")
    placedb.debug_str()

    node_pos = {}
    for node in placedb.node_info:
        node_pos[node] = placedb.node_info[node]["x"], placedb.node_info[node]["y"]

    hpwl, cost = comp_res(placedb, node_pos, 1.0)
    print("hpwl:{}  cost:{}".format(hpwl, cost))

