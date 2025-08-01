##
import json
import os
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image
from agent_memory import AgentMemory

from LLM.LLM_oppo import LLM_oppo

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CELL_SIZE = 0.125
ANGLE = 15


class lm_agent_oppo:
    """
    大模型驱动的智能体类
    主要功能：
    1. 使用大模型进行决策规划
    2. 处理环境观察和状态
    3. 执行导航和物体操作
    4. 管理智能体记忆和状态
    """

    def __init__(self, agent_id, logger, max_frames, args, output_dir="results"):
        """
        初始化大模型智能体

        参数:
            agent_id: 智能体ID
            logger: 日志记录器
            max_frames: 最大帧数
            args: 配置参数
            output_dir: 输出目录
        """
        # 环境状态相关变量
        self.with_oppo = None  # 对手持有的物体
        self.oppo_pos = None  # 对手位置
        self.with_character = None  # 角色持有的物体
        self.color2id = None  # 颜色到ID的映射
        self.satisfied = None  # 已完成的物体
        self.object_list = None  # 物体列表
        self.container_held = None  # 持有的容器
        self.gt_mask = None  # 是否使用真实掩码

        # 物体信息存储
        self.object_info = (
            {}
        )  # 物体详细信息 {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = (
            {}
        )  # 每个房间的物体 {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}

        # 地图相关
        self.id_map = None  # ID地图
        self.object_map = None  # 物体地图

        # 智能体基本信息
        self.agent_id = agent_id
        self.agent_type = "lm_agent_oppo"
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id

        # 环境配置
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True

        # 场景边界
        self._scene_bounds = {"x_min": -15, "x_max": 15, "z_min": -7.5, "z_max": 7.5}

        # 导航参数
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.logger = logger
        random.seed(1024)
        self.debug = True

        # 观察和状态相关
        self.new_object_list = None  # 新发现的物体
        self.visible_objects = None  # 可见物体
        self.num_frames = None  # 当前帧数
        self.steps = None  # 步数
        self.obs = None  # 当前观察
        self.local_step = 0  # 局部步数

        # 动作历史
        self.last_action = None  # 上一个动作
        self.pre_action = None  # 前一个动作

        # 目标相关
        self.goal_objects = None  # 目标物体
        self.dropping_object = None  # 正在放置的物体

        # 大模型配置
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM_oppo(
            self.source,
            self.lm_id,
            self.prompt_template_path,
            self.communication,
            self.cot,
            self.args,
            self.agent_id,
        )
        self.action_history = []  # 动作历史
        self.dialogue_history = []  # 对话历史
        self.plan = None  # 当前计划

        # 房间和位置相关
        self.rooms_name = None  # 房间名称
        self.rooms_explored = {}  # 已探索的房间
        self.position = None  # 当前位置
        self.forward = None  # 朝向
        self.current_room = None  # 当前房间
        self.holding_objects_id = None  # 持有的物体ID
        self.oppo_holding_objects_id = None  # 对手持有的物体ID
        self.oppo_last_room = None  # 对手最后所在的房间
        self.rotated = None  # 旋转状态
        self.navigation_threshold = 5  # 导航阈值
        self.detection_threshold = 5  # 检测阈值

        # 通信相关配置
        self.communication = args.communication  # 是否启用通信功能
        print(f"是否启用通信：{self.communication}")
        self.dialogue_history = []  # 存储对话历史记录，用于记录智能体之间的通信内容
        self.agent_names = ["Alice", "Bob"]  # 智能体名称列表，用于标识消息发送者

    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        """
        获取指定颜色的点云数据

        参数:
            color: 目标颜色

        返回:
            点云数据
        """
        depth = self.obs["depth"].copy()
        for i in range(len(self.obs["seg_mask"])):
            for j in range(len(self.obs["seg_mask"][0])):
                if (self.obs["seg_mask"][i][j] != color).any():
                    depth[i][j] = 1e9
        # camera info
        FOV = self.obs["FOV"]
        W, H = depth.shape
        cx = W / 2.0
        cy = H / 2.0
        fx = cx / np.tan(math.radians(FOV / 2.0))
        fy = cy / np.tan(math.radians(FOV / 2.0))

        # Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs["camera_matrix"]
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        """
        计算物体位置

        参数:
            o_dict: 物体信息字典

        返回:
            物体位置坐标
        """
        pc = self.get_pc(o_dict["seg_color"])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o["type"] is not None and o["type"] < 4:
                visible_obj.append(o)
        return visible_obj

    def get_object_list(self):
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]
                if (
                    id in self.satisfied
                    or id in self.holding_objects_id
                    or id in self.oppo_holding_objects_id
                    or self.object_info[id] in object_list[object_type]
                ):
                    continue
                object_list[object_type].append(self.object_info[id])
                room = self.env_api["belongs_to_which_room"](
                    self.object_info[id]["position"]
                )
                if room is None:
                    self.logger.warning(f"obj {self.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[id])
        self.object_list = object_list

    def get_new_object_list(self):
        self.visible_objects = self.obs["visible_objects"]
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict["id"] is None:
                continue
            self.color2id[o_dict["seg_color"]] = o_dict["id"]
            if (
                o_dict["id"] is None
                or o_dict["id"] in self.satisfied
                or o_dict["id"] in self.with_character
                or o_dict["type"] == 4
            ):
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict["id"]
            new_obj = False
            if object_id not in self.object_info:
                self.object_info[object_id] = {}
                new_obj = True
            self.object_info[object_id]["id"] = object_id
            self.object_info[object_id]["type"] = o_dict["type"]
            self.object_info[object_id]["name"] = o_dict["name"]
            if o_dict["type"] == 3:  # the agent
                if o_dict["id"] == self.opponent_agent_id:
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api["belongs_to_which_room"](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue
            if object_id in self.satisfied or object_id in self.with_character:
                continue
            self.object_info[object_id]["position"] = position
            if o_dict["type"] == 0:
                x, y, z = self.object_info[object_id]["position"]

                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[0].append(object_id)

            elif o_dict["type"] == 1:
                x, y, z = self.object_info[object_id]["position"]
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[1].append(object_id)
            elif o_dict["type"] == 2:
                x, y, z = self.object_info[object_id]["position"]
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[2].append(object_id)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any():
                return -100  # wall
            else:
                return self.agent_id  # agent
        else:
            return self.color2id[color]

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold=1.0):
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith("transport"):
            if self.env_api["belongs_to_which_room"](
                np.array([x, 0, z])
            ) != self.env_api["belongs_to_which_room"](np.array([gx, 0, gz])):
                return False
        return d < threshold

    def reset(
        self,
        obs,
        goal_objects=None,
        output_dir=None,
        env_api=None,
        rooms_name=None,
        agent_color=[-1, -1, -1],
        agent_id=0,
        gt_mask=True,
        save_img=True,
    ):
        self.force_ignore = []
        self.agent_memory = AgentMemory(
            agent_id=self.agent_id,
            agent_color=agent_color,
            output_dir=output_dir,
            gt_mask=self.gt_mask,
            gt_behavior=True,
            env_api=env_api,
            constraint_type=None,
            map_size=self.map_size,
            scene_bounds=self._scene_bounds,
        )
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum([v for k, v in goal_objects.items()])
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        # print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api["belongs_to_which_room"](self.position)
        self.rotated = None
        self.rooms_explored = {}

        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.gt_mask = gt_mask
        if self.gt_mask == True:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection

            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5
        # print(self.rooms_name)
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)
        return action

    def gotoroom(self):
        target_room = " ".join(self.plan.split(" ")[2:4])
        if target_room[-1] == ",":
            target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api["center_of_room"](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # add an interruption if anything new happens
        if (
            len(self.new_object_list[0])
            + len(self.new_object_list[1])
            + len(self.new_object_list[2])
            > 0
        ):
            #更新动作历史
            self.action_history[-1] = self.action_history[-1].replace(
                self.plan, f"go to {self.current_room}"
            )
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
            return None
        return self.move(target_pos)

    def goexplore(self):
        target_room = " ".join(self.plan.split(" ")[-2:])
        # assert target_room == self.current_room, f"{target_room} != {self.current_room}"
        target_pos = self.env_api["center_of_room"](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.roatated = 0
            self.rooms_explored[target_room] = "all"
            self.plan = None
            return None
        self.rotated += 1
        action = {"type": 1}
        return action

    def gograsp(self):
        target_object_id = int(self.plan.split(" ")[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None

        if self.target_pos is None:
            self.target_pos = copy.deepcopy(
                self.object_info[target_object_id]["position"]
            )
        target_object_pos = self.target_pos

        if (
            target_object_id not in self.object_info
            or target_object_id in self.with_oppo
        ):
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {
            "type": 3,
            "object": target_object_id,
            "arm": "left" if self.obs["held_objects"][0]["id"] is None else "right",
        }
        return action

    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]["position"])
        target_pos = self.target_pos

        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)
        if self.obs["held_objects"][0]["type"] is not None:
            self.dropping_object += [self.obs["held_objects"][0]["id"]]
            if self.obs["held_objects"][0]["type"] == 1:
                self.dropping_object += [
                    x for x in self.obs["held_objects"][0]["contained"] if x is not None
                ]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]["id"]]
            if self.obs["held_objects"][1]["type"] == 1:
                self.dropping_object += [
                    x for x in self.obs["held_objects"][1]["contained"] if x is not None
                ]
            return {"type": 5, "arm": "right"}

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action

    def detect(self):
        """
        执行目标检测

        返回:
            obj_infos: 检测到的物体信息
            curr_seg_mask: 分割掩码
        """
        detect_result = self.detection_model(self.obs["rgb"][..., [2, 1, 0]])[
            "predictions"
        ][0]
        obj_infos = []
        curr_seg_mask = np.zeros(
            (self.obs["rgb"].shape[0], self.obs["rgb"].shape[1], 3)
        ).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result["labels"])):
            if detect_result["scores"][i] < 0.3:
                continue
            mask = detect_result["masks"][:, :, i]
            label = detect_result["labels"][i]
            curr_info = self.env_api["get_id_from_mask"](
                mask=mask, name=self.detection_model.cls_to_name_map(label)
            ).copy()
            if curr_info["id"] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info["seg_color"]
        curr_with_seg, curr_seg_flag = self.env_api["get_with_character_mask"](
            character_object_ids=self.with_character
        )
        curr_seg_mask = curr_seg_mask * (
            ~np.expand_dims(curr_seg_flag, axis=-1)
        ) + curr_with_seg * np.expand_dims(curr_seg_flag, axis=-1)
        return obj_infos, curr_seg_mask

    def LLM_plan(self):
        """
        使用大模型进行规划，包括通信决策

        返回:
            plan: 规划结果，可能包含通信动作
            a_info: 规划信息
        """
        # 将对话历史作为上下文输入传递给大模型
        # 这样大模型可以根据历史对话内容做出更合理的决策
        return self.LLM.run(
            self.num_frames,
            self.current_room,
            self.rooms_explored,
            self.obs["held_objects"],
            [self.object_info[x] for x in self.satisfied if x in self.object_info],
            self.object_list,
            self.object_per_room,
            self.action_history,
            self.dialogue_history,  # 对话历史作为上下文输入
            self.obs["oppo_held_objects"],
            self.oppo_last_room,
        )

    def act(self, obs):
        """
        执行动作

        参数:
            obs: 环境观察 文本形式

        返回:
            action: 要执行的动作
        """
        self.obs = obs.copy()
        self.obs["rgb"] = self.obs["rgb"].transpose(1, 2, 0)
        self.num_frames = obs["current_frames"]
        self.steps += 1

        if not self.gt_mask:
            self.obs["visible_objects"], self.obs["seg_mask"] = self.detect()

        #无效动作处理
        if obs["valid"] == False:
            if self.last_action is not None and "object" in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.id_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.satisfied.append(self.last_action["object"])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"

        # 处理通信消息
        if self.communication:

            # 遍历所有接收到的消息
            for i in range(len(obs["messages"])): #长度为2 Alice+Bob
                if obs["messages"][i] is not None:
                    # 将消息添加到对话历史中，格式为"智能体名称: 消息内容"
                    # 使用copy.deepcopy确保消息内容不会被意外修改
                    self.dialogue_history.append(
                        f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}"
                    )

        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        # 更新当前房间
        current_room = self.env_api["belongs_to_which_room"](self.position)
        if current_room is not None:                                                                                                                                            
            self.current_room = current_room
        
        self.room_distance = self.env_api["get_room_distance"](self.position)
        if (
            self.current_room not in self.rooms_explored
            or self.rooms_explored[self.current_room] != "all"
        ):
            self.rooms_explored[self.current_room] = "part"

        if self.agent_id not in self.with_character:
            self.with_character.append(
                self.agent_id
            )  # DWH: buggy env, need to solve later.
        #查询手中的物体
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs["held_objects"]: #更新
            if x["type"] == 0:
                self.holding_objects_id.append(x["id"])
                if x["id"] not in self.with_character:
                    self.with_character.append(
                        x["id"]
                    )  # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x["type"] == 1:
                self.holding_objects_id.append(x["id"])
                if x["id"] not in self.with_character:
                    self.with_character.append(
                        x["id"]
                    )  # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
                for y in x["contained"]:
                    if y is None:
                        break
                    if y not in self.with_character:
                        self.with_character.append(y)
                    # self.with_character.append(y)
        oppo_name = {}
        oppo_type = {}
        for x in self.obs["oppo_held_objects"]:
            if x["type"] == 0:
                self.oppo_holding_objects_id.append(x["id"])
                self.with_oppo.append(x["id"])
                oppo_name[x["id"]] = x["name"]
                oppo_type[x["id"]] = x["type"]
            elif x["type"] == 1:
                self.oppo_holding_objects_id.append(x["id"])
                self.with_oppo.append(x["id"])
                oppo_name[x["id"]] = x["name"]
                oppo_type[x["id"]] = x["type"]
                for i, y in enumerate(x["contained"]):
                    if y is None:
                        break
                    self.with_oppo.append(y)
                    oppo_name[y] = x["contained_name"][i]
                    oppo_type[y] = 0
        for obj in self.with_oppo:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_info[obj] = {
                    "name": oppo_name[obj],
                    "id": obj,
                    "type": oppo_type[obj],
                }
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0
        if not self.obs["valid"]:  # invalid, the object is not there
            if self.last_action is not None and "object" in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action["object"])] = 0
                self.id_map[np.where(self.id_map == self.last_action["object"])] = 0
        if len(self.dropping_object) > 0 and self.obs["status"] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            self.satisfied += self.dropping_object
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]##
        temp_with_oppo = []##
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x is None or x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])## temp with oppo?
            if "contained" in x:##contain structure
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied
        #更新记忆
        self.agent_memory.update(
            obs,
            ignore_ids=ignore_ids,
            ignore_obstacles=ignore_obstacles,
            save_img=self.save_img,
        )

        if self.obs["status"] == 0:  # ongoing 这里就不调用了
            return {"type": "ongoing"}

        self.get_new_object_list()
        print(self.new_object_list)
        self.get_object_list()

        info = {
            "satisfied": self.satisfied,
            "object_list": self.object_list,
            "new_object_list": self.new_object_list,
            "current_room": self.current_room,
            "visible_objects": self.filtered(self.obs["visible_objects"]),
            "obs": {
                k: v
                for k, v in self.obs.items()
                if k
                not in ["rgb", "depth", "seg_mask", "camera_matrix", "visible_objects"]
            },
        }

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_pos = None
                if lm_times > 0:
                    print(info)
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")
                plan, a_info = self.LLM_plan()
                if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                self.plan = plan
                self.action_history.append(
                    f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}"
                )
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
            if self.plan.startswith("go to"):
                action = self.gotoroom()
            elif self.plan.startswith("explore"):
                self.explore_count = 0
                action = self.goexplore()
            elif self.plan.startswith("go grasp"):
                action = self.gograsp()
            elif self.plan.startswith("put"):
                action = self.putin()
            elif self.plan.startswith("transport"):
                action = self.goput() #high-level action
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith("send a message"):
                # 发送消息动作
                action = {
                    "type": 6,  # 动作类型6表示发送消息
                    "message": " ".join(
                        self.plan.split(" ")[3:]
                    ),  # 提取消息内容，去掉"send a message:"前缀
                }
                self.plan = None  # 清除当前计划，准备执行下一个动作
            elif self.plan.startswith("wait"):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        info.update({"action": action, "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
            # print(
            #     f"{self.agent_names}当前的动作和计划是 action: {action}, plan: {self.plan}"
            # )
        self.last_action = action
        return action

    

    def visualize_semantic_map(self, save_path=None):
        """
        可视化语义地图（object_map），不同类型用不同颜色显示，并可选保存到文件
        """
        # 定义颜色映射：0-空地，1-普通物体，2-容器，3-目标物体
        color_map = {
            0: [255, 255, 255],  # 白色-空地
            1: [0, 255, 0],      # 绿色-普通物体
            2: [0, 0, 255],      # 蓝色-容器
            3: [255, 0, 0],      # 红色-目标物体
        }
        h, w = self.object_map.shape
        vis_map = np.zeros((h, w, 3), dtype=np.uint8)
        for k, color in color_map.items():
            vis_map[self.object_map == k] = color

        plt.figure(figsize=(10, 5))
        plt.imshow(vis_map)
        plt.title("Semantic Map (object_map)")
        # 图例
        patches = [
            mpatches.Patch(color=np.array(color_map[k])/255, label=f"type {k}")
            for k in color_map
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # 用法示例（在lm_agent_oppo对象中调用）：
    # self.visualize_semantic_map("semantic_map.png")
