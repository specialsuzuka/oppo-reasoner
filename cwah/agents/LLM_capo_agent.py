from LLM.LLM_capo import LLM_capo
from agents.LLM_agent import LLM_agent

class capo_agent(LLM_agent):
    '''
    from LLM_agent
    '''
    def __init__ (self,agent_id, char_index, args):
        super().__init__(agent_id,char_index,args)
        self.LLM = LLM_capo(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
        self.subplan = None
        self.metaplan = None
        self.host = 2-self.agent_id
        self.oppo_progress = ""
        self.node_memory = []
        #counting 
        self.comm_num = 0
        self.characters = 0

    def goexplore(self):
        target_room_id = int(self.subplan.split(' ')[-1][1:-1])
        if self.current_room['id'] == target_room_id:
            self.subplan = None
            return None
        return self.subplan.replace('[goexplore]', '[walktowards]')

    def gocheck(self):
        assert len(self.grabbed_objects) < 2 # must have at least one free hands
        target_container_id = int(self.subplan.split(' ')[-1][1:-1])
        target_container_name = self.subplan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]
        if 'OPEN' in target_container['states']:
            self.subplan = None
            return None
        if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.subplan.replace('[gocheck]', '[open]') # conflict will work right?
        else:
            return self.subplan.replace('[gocheck]', '[walktowards]')
        
    def gograb(self):
        target_object_id = int(self.subplan.split(' ')[-1][1:-1])
        target_object_name = self.subplan.split(' ')[1]
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.subplan = None
            return None
        assert len(self.grabbed_objects) < 2 # must have at least one free hands

        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.subplan = None
            return None
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.subplan.replace('[gograb]', '[grab]')
        else:
            return self.subplan.replace('[gograb]', '[walktowards]')

    def goput(self):
        if len(self.grabbed_objects) == 0:
            self.subplan = None
            return None
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and'CLOSED' in y['states']:
                return self.subplan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"

    def LLM_metaplan_init(self):
        output = self.LLM.meta_plan_init()
        self.characters += len(output.split(""))
        self.comm_num += 1
        return output
    def LLM_disscuss_refine(self,
                            refine):
        output = self.LLM.disscuss_refine(refine,
                                          self.metaplan,
                                          self.oppo_progress,
                                          self.current_room,
                                          [self.id2node[x] for x in self.grabbed_objects],
                                          self.satisfied,
                                          self.unchecked_containers,
                                          self.ungrabbed_objects,
                                          self.id_inside_room[self.goal_location_id],
                                          self.action_history,
                                          self.dialogue_history,
                                          self.opponent_grabbed_objects,
                                          self.id_inside_room[self.opponent_agent_id]
                                          )
        self.characters += len(output.split(""))
        self.comm_num += 1
        return output
    def LLM_parsing(self):
        output = self.LLM.parsing(self.metaplan,
                                  self.current_room,
                                  [self.id2node[x] for x in self.grabbed_objects],
                                  self.satisfied,
                                  self.unchecked_containers,
                                  self.ungrabbed_objects,
                                  self.id_inside_room[self.goal_location_id],
                                  self.action_history,
                                  self.dialogue_history,
                                  self.opponent_grabbed_objects,
                                  self.id_inside_room[self.opponent_agent_id]
                                  )
        self.characters += len(output.split(""))
        return output
    
    def LLM_progress_sending(self):
        output = self.LLM.progress_sending(
            self.current_room,
            [self.id2node[x] for x in self.grabbed_objects],
            self.unchecked_containers,
            self.ungrabbed_objects,
            self.id_inside_room[self.goal_location_id],
            self.satisfied,
            self.opponent_grabbed_objects,
             self.id_inside_room[self.opponent_agent_id]
        )
        return output
    
    def get_action(self,observation, goal):
        def updater():
            visable_node = []
            for node in observation["nodes"]:
                visable_node.append(node["class_name"])
            new_node = []
            for node in visable_node:
                if node not in self.node_memory:
                    new_node.append(node)
            for node in new_node:
                self.node_memory.append(node)
        for i in range(len(observation["messages"])):
            if observation["messages"][i] is not None:
                self.dialogue_history.append(f"{self.agent_names[i + 1]}: {observation['messages'][i]}")
        satisfied, unsatisfied = self.check_progress(observation, goal)
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied#{'on_pudding_<coffeetable> (268)': 1, 'on_juice_<coffeetable> (268)': 1, 'on_apple_<coffeetable> (268)': 1, 'on_cupcake_<coffeetable> (268)': 2}
            self.satisfied = satisfied

        # add target object
        target_objects = []
        for target in list(unsatisfied.keys()):
            if unsatisfied[target] != 0:
                target_object = target.split("_")[1]
                target_objects.append(target_object)
        
        # update the node_memory
       

        #nodes in observation:[{'id': 211, 'category': 'Furniture', 'class_name': 'bookshelf', 'prefab_name': 'PRE_FUR_Bookshelf_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 212, 'category': 'Furniture', 'class_name': 'chair', 'prefab_name': 'PRE_FUR_Kitchen_chair_01_03', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 213, 'category': 'Furniture', 'class_name': 'desk', 'prefab_name': 'PRE_FUR_CPU_table_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 214, 'category': 'Furniture', 'class_name': 'nightstand', 'prefab_name': 'PRE_FUR_Nightstand_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 215, 'category': 'Furniture', 'class_name': 'bed', 'prefab_name': 'PRE_FUR_Bed_01_01_03', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 216, 'category': 'Furniture', 'class_name': 'cabinet', 'prefab_name': 'Cabinet_1', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 217, 'category': 'Electronics', 'class_name': 'lightswitch', 'prefab_name': 'PRE_ELE_Light_switch_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 218, 'category': 'Electronics', 'class_name': 'powersocket', 'prefab_name': 'PRE_ELE_Power_socket_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 219, 'category': 'Electronics', 'class_name': 'radio', 'prefab_name': 'Radio_6', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 241, 'category': 'Props', 'class_name': 'dishbowl', 'prefab_name': 'FMGP_PRE_Wooden_bowl_1024', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 242, 'category': 'Props', 'class_name': 'box', 'prefab_name': 'FMGP_PRE_Wooden_box_1024', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 245, 'category': 'Props', 'class_name': 'plate', 'prefab_name': 'PRE_PRO_Plate_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 248, 'category': 'Props', 'class_name': 'cellphone', 'prefab_name': 'Cellphone_6a', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 249, 'category': 'Doors', 'class_name': 'doorjamb', 'prefab_name': 'PRE_DOO_Doorjamb_04', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 250, 'category': 'Doors', 'class_name': 'door', 'prefab_name': 'PRE_DOO_Door_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 251, 'category': 'Decor', 'class_name': 'rug', 'prefab_name': 'PRE_DEC_Rug_01_09', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 252, 'category': 'Decor', 'class_name': 'pillow', 'prefab_name': 'HSHP_PRE_DEC_Pillow_01_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 253, 'category': 'Decor', 'class_name': 'pillow', 'prefab_name': 'HSHP_PRE_DEC_Pillow_01_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 254, 'category': 'Lamps', 'class_name': 'ceilinglamp', 'prefab_name': 'PRE_LAM_Ceiling_lamp_02_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 255, 'category': 'Floor', 'class_name': 'floor', 'prefab_name': 'PRE_FLO_Wood_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 256, 'category': 'Floor', 'class_name': 'floor', 'prefab_name': 'PRE_FLO_Wood_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 257, 'category': 'Floor', 'class_name': 'floor', 'prefab_name': 'PRE_FLO_Wood_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 258, 'category': 'Floor', 'class_name': 'floor', 'prefab_name': 'PRE_FLO_Wood_01_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 259, 'category': 'Ceiling', 'class_name': 'ceiling', 'prefab_name': 'PRE_CEI_Paint_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 260, 'category': 'Ceiling', 'class_name': 'ceiling', 'prefab_name': 'PRE_CEI_Paint_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 261, 'category': 'Ceiling', 'class_name': 'ceiling', 'prefab_name': 'PRE_CEI_Paint_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 262, 'category': 'Ceiling', 'class_name': 'ceiling', 'prefab_name': 'PRE_CEI_Paint_01_05', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 263, 'category': 'Walls', 'class_name': 'wall', 'prefab_name': 'PRE_WAL_WAL_Doorway_corner_01_02_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 264, 'category': 'Walls', 'class_name': 'wall', 'prefab_name': 'PRE_WAL_WAL_Corner_02_02_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 265, 'category': 'Walls', 'class_name': 'wall', 'prefab_name': 'PRE_WAL_WAL_Corner_02_02_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 266, 'category': 'Walls', 'class_name': 'wall', 'prefab_name': 'PRE_WAL_WAL_Doorway_corner_01_02_02', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 1, 'category': 'Characters', 'class_name': 'character', 'prefab_name': 'Female1', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 11, 'category': 'Rooms', 'class_name': 'kitchen', 'prefab_name': 'PRE_ROO_Kitchen_01', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 172, 'category': 'Rooms', 'class_name': 'bathroom', 'prefab_name': 'PRE_ROO_Bathroom_03', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 210, 'category': 'Rooms', 'class_name': 'bedroom', 'prefab_name': 'PRE_ROO_Bedroom_03', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}, {'id': 267, 'category': 'Rooms', 'class_name': 'livingroom', 'prefab_name': 'PRE_ROO_Livingroom_08', 'obj_transform': {...}, 'bounding_box': {...}, 'properties': [...], 'states': [...]}]
        obs = self.filter_graph(observation)

        if obs["progress"][self.host] is not None:
            self.oppo_progress = obs["progress"][self.host]

        if (not self.host) and obs["metaplan"][0] is not None:
            self.metaplan = obs["metaplan"][0]


        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in obs['nodes']}
        for e in obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r == 'INSIDE':
                    self.current_room = self.id2node[y]
                elif r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []
        for x in obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
                ungrabbed_objects.append(x)

        if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': obs,
                "obs": {
                            "grabbed_objects": self.grabbed_objects,
                            "opponent_grabbed_objects": opponent_grabbed_objects,
                            "reachable_objects": self.reachable_objects,
                            "progress": {
                                "unchecked_containers": self.unchecked_containers,
                                "ungrabbed_objects": self.ungrabbed_objects,
                                        },
                        "satisfied": self.satisfied,
                        "current_room": self.current_room['class_name'],
                        },
                }
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects
        
        if obs["call_for_disscussion"] == 1:
            return "[wait_for_disscussion]",{}

    
        if obs["ep_id"] == 0:##TODO:check the init process
            if self.host:
                metaplan = self.LLM_metaplan_init()
                self.metaplan = metaplan
                action = "[metaplan]" + "<" + metaplan + ">"
                self.action_history.append("[init_metaplan]")
                
            else:
                action = "[waiting]"
            updater()
            
            return action, info

        if obs["disscussion"] == 1 and obs["turns"] == 0:
            progress = self.LLM_progress_sending()
            action = "[progress]" + "<" + progress + ">"
            self.action_history.append("[disscussion]")
            updater()
            return action,info
        
        if obs['disscussion'] == 1 and obs["turns"] == 1:
            if self.host:
                metaplan = self.LLM_disscuss_refine(1)
                self.metaplan = metaplan
                action = "[metaplan]" + "<" + metaplan + ">"
                
            else:
                action = "[waiting]"
            updater()
            return action,info
        
        if obs["disscussion"] == 1 and obs["turns"] == 2:#TODO:check if these messages is sent in the messages
            if self.host:
                message = self.LLM_disscuss_refine(0)
                action = "[send_message1]" + "<" + message + ">"

            else:
                action = "[waiting]"
            updater()
            return action,info 
        
        if obs['disscussion'] == 1 and obs["turns"] == 3:
            if not self.host:
                message = self.LLM_disscuss_refine(0)
                action = "[send_message2]" + "<" + message + ">"

            else:
                action = "[waiting]"
            updater()
            return action,info 
        
        
                
        #create a visable node
        visable_node = []
        for node in observation["nodes"]:
            visable_node.append(node["class_name"])
        
        #new node
        new_node = []
        for node in visable_node:
            if node not in self.node_memory:
                new_node.append(node)

        for target in target_objects:
            for node in new_node:
                if target in node:
                    action = "[wait_for_disscussion]"
                    return action , info
            
        ## trigger 1 refine the function
        # for target in target_objects:
        #     for item in self.node_memory:
        #         if target in item:
        #             action = "[wait_for_disscussion]"
        #             return action,info
        #update the memory node
        for node in new_node:
            self.node_memory.append(node)
        # for node in observation["nodes"]:
        #     if node["class_name"] not in self.node_memory:
        #         self.node_memory.append(node["class_name"])



        action = None
        while action is None:
            if self.subplan is None:
                subplan = self.LLM_parsing()
                self.subplan = subplan
                self.action_history.append(self.subplan)

            if self.subplan.startswith('[goexplore]'):
                action = self.goexplore()
            elif self.subplan.startswith('[gocheck]'):
                action = self.gocheck()
            elif self.subplan.startswith('[gograb]'):
                action = self.gograb()
            elif self.subplan.startswith('[goput]'):
                action = self.goput()
            elif self.subplan.startswith('[wait]'):##TODO:change to waiting
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")
        
        
        self.steps += 1
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        # self.last_location = self.location
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.subplan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0

        return action, info
    
    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        super().reset(obs,
                      containers_name,
                      goal_objects_name,
                      rooms_name,
                      room_info,
                      goal)
        self.subplan = None
        self.oppo_progress = "" 
        self.metaplan = None
        self.node_memory = []
        self.comm_num = 0
        self.characters = 0
        self.LLM.api = 0
        self.LLM.tokens = 0
    def get_api(self):
        return self.LLM.api
    def get_tokens(self):
        return self.LLM.tokens
    def filter_graph(self, obs):
        relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
        relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
        new_graph = {
            "edges": [edge for edge in obs['edges'] if
                        edge['from_id'] in relative_id and edge['to_id'] in relative_id],
            "nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
        }
        
        if "progress" in list(obs.keys()) and obs["progress"] is not None:
            new_graph["progress"] = obs["progress"]
        if "metaplan" in list(obs.keys()) and obs["metaplan"] is not None:
            new_graph["metaplan"] = obs["metaplan"]
        if "call_for_disscussion" in list(obs.keys()) and obs["call_for_disscussion"] is not None:
            new_graph["call_for_disscussion"] = obs["call_for_disscussion"]
        if "ep_id" in list(obs.keys()) and obs["ep_id"] is not None:
            new_graph["ep_id"] = obs["ep_id"]
        if "disscussion" in list(obs.keys()) and obs["disscussion"] is not None:
            new_graph["disscussion"] = obs["disscussion"]
        if "turns" in list(obs.keys()) :
            new_graph["turns"] = obs["turns"]
        if "round"in list(obs.keys()):
            new_graph["round"] = obs["round"]

        return new_graph




