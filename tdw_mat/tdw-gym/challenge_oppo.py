##
import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys

# add this dictionary to python env path:
base_path = os.getcwd()
sys.path.append(base_path)

from h_agent import H_agent
from lm_agent_oppo import lm_agent_oppo
from lm_agent import lm_agent

# 注册测试环境
gym.envs.registration.register(id="transport_challenge_MA", entry_point="tdw_gym:TDW")


class Challenge_oppo:
    """
    多智能体运输挑战环境管理类
    用于管理环境、执行评估和记录结果
    """

    def __init__(
        self,
        logger,
        time_logger,
        port,
        data_path,
        output_dir,
        number_of_agents=2,
        max_frames=3000,
        launch_build=True,
        screen_size=256,
        data_prefix="dataset/nips_dataset/",
        gt_mask=True,
        save_img=True,
    ):
        """
        初始化挑战环境

        参数:
            logger: 日志记录器
            port: 环境端口号
            data_path: 数据文件路径
            output_dir: 输出目录
            number_of_agents: 智能体数量（默认2个）
            max_frames: 最大帧数（默认3000）
            launch_build: 是否启动构建（默认True）
            screen_size: 屏幕大小（默认512）
            data_prefix: 数据集前缀路径
            gt_mask: 是否使用真实掩码（默认True）
            save_img: 是否保存图像（默认True）
        """
        self.env = gym.make(
            "transport_challenge_MA",
            port=port,
            number_of_agents=number_of_agents,
            save_dir=output_dir,
            max_frames=max_frames,
            launch_build=launch_build,
            screen_size=screen_size,
            data_prefix=data_prefix,
            gt_mask=gt_mask,
        )
        self.gt_mask = gt_mask
        self.logger = logger
        self.time_logger = time_logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.save_img = save_img
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")

    def submit(self, agents, logger, eval_episodes):
        """
        执行智能体评估过程

        参数:
            agents: 智能体列表
            logger: 日志记录器
            eval_episodes: 要评估的回合列表

        返回:
            float: 平均完成率
        """

        total_finish = 0.0
        if eval_episodes[0] == -1:
            eval_episodes = range(len(self.data))
        num_eval_episodes = len(eval_episodes)
        # 无循环部分
        start = time.time()
        results = {}
        #all episode charaters
        total_0_charaters = 0
        total_1_charaters = 0
        total_0_com = 0
        total_1_com = 0
        for i, episode in enumerate(eval_episodes):

            #characters per episode
            episode_0_charaters = 0
            episode_1_charaters = 0
            episode_0_com = 0
            episode_1_com = 0
            print(f"当前执行的episode为：{episode}")
            start_time = time.time()

            # 检查是否已经评估过该回合
            if os.path.exists(
                os.path.join(self.output_dir, str(episode), "result_episode.json")
            ):
                with open(
                    os.path.join(self.output_dir, str(episode), "result_episode.json"),
                    "r",
                ) as f:
                    result = json.load(f)
                total_finish += result["finish"] / result["total"]
                results[episode] = result
                continue
            # The episode has been evaluated before

            # 创建输出目录
            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info(
                "Episode {} ({}/{})".format(episode, i + 1, num_eval_episodes)
            )
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")

            # 重置环境
            state, info, env_api = self.env.reset(
                seed=self.data[episode]["seed"],
                options=self.data[episode],
                output_dir=os.path.join(self.output_dir, str(episode)),
            )

            # 重置每个智能体
            for id, agent in enumerate(agents):
                if type(env_api) == list:
                    curr_api = env_api[id]
                else:
                    curr_api = env_api
                if info["goal_description"] is not None:
                    if agent.agent_type == "h_agent":
                        agent.reset(
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                        )
                    elif agent.agent_type == "lm_agent_oppo":
                        agent.reset(
                            obs=state[str(id)],
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            rooms_name=info["rooms_name"],
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                        )
                    elif agent.agent_type == "lm_agent":
                        agent.reset(
                            obs=state[str(id)],
                            goal_objects=info["goal_description"],
                            output_dir=os.path.join(self.output_dir, str(episode)),
                            env_api=curr_api,
                            agent_color=info["agent_colors"][id],
                            agent_id=id,
                            rooms_name=info["rooms_name"],
                            gt_mask=self.gt_mask,
                            save_img=self.save_img,
                        )
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir=os.path.join(self.output_dir, str(episode)))
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")

            # 执行评估过程
            episode_start_time = time.time()  # 记录本episode总计时
            act_total_time = 0.0  # 记录act方法总时间
            act_num = 0
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            
            local_reward = 0.0
            while not done:
                step_num += 1
                actions = {}
                # 保存图片
                if self.save_img:
                    self.env.save_images(
                        os.path.join(self.output_dir, str(episode), "Images")
                    )
                for agent_id, agent in enumerate(agents):
                    act_start = time.time()
                    actions[str(agent_id)] = agent.act(state[str(agent_id)])
                    act_end = time.time()
                    act_num += 1
                    act_total_time += (act_end - act_start)
                    print(actions[str(agent_id)]['type'],"characters",agent.get_tokens(),"com_num:",agent.get_com_cost())
                    # if actions[str(agent_id)]['type'] == 6 and agent_id ==0:
                    #     communication_num_0 += 1
                    #     print("Communication action taken by agent:", agent.agent_names[agent.agent_id])
                    # if actions[str(agent_id)]['type'] == 6 and agent_id ==1:
                    #     communication_num_1 += 1
                    #     print("Communication action taken by agent:", agent.agent_names[agent.agent_id])
                    print(f"agent_name:{agent.agent_names[agent.agent_id]}, action: {actions[str(agent_id)]}\n")
                    #这个地方
                state, reward, done, info = self.env.step(actions)
                local_reward += reward
                local_finish = self.env.check_goal()
                self.logger.info(
                    f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}"
                )
                if done:
                    break
            #episode count
            episode_0_charaters = agents[0].get_tokens()
            episode_1_charaters = agents[1].get_tokens()
            episode_0_com = agent[0].get_com_cost()
            episode_1_com = agent[1].get_com_cost()

            #total count
            total_0_charaters += episode_0_charaters
            total_1_charaters += episode_1_charaters
            total_0_com += episode_0_com
            total_1_com += episode_1_com

            episode_total_time = time.time() - episode_start_time
            self.time_logger.info(f"Episode {episode} total time: {episode_total_time:.4f} secs")
            self.time_logger.info(f"Episode {episode} total act() time: {act_total_time:.4f} secs")

            # 记录结果
            total_finish += local_finish[0] / local_finish[1]
            result = {
                "finish": local_finish[0],
                "total": local_finish[1],
                "step_num": step_num,
                "frame": self.env.num_frames,
                "communication num_0": episode_0_com,
                "communication num_1": episode_1_com,
                "charater_0":episode_0_charaters,
                "charater_1":episode_1_charaters,
                "episode_total_time": episode_total_time,
                "act_total_time": act_total_time,
                "act_num": act_num
            }
            with open(
                os.path.join(self.output_dir, str(episode), "result_episode.json"), "w"
            ) as f:
                json.dump(result, f)
            results[episode] = result

        # 计算并保存最终结果
        avg_finish = total_finish / num_eval_episodes
        results = {"episode_results": results, "avg_finish": avg_finish}
        with open(os.path.join(self.output_dir, "eval_result.json"), "w") as f:
            json.dump(results, f, indent=4)

        #whole results
        with open("./count_results/counts.txt","a+") as f:
            f.write(f"time:{time.time()}")
            f.write(f"total_characters:{total_0_charaters+total_1_charaters}")
            f.write(f"total_0_characters:{total_0_charaters}")
            f.write(f"total_1_characters:{total_1_charaters}")
            f.write(f"total_0_com:{total_0_com}")
            f.write(f"total_1_com:{total_1_com}")
            f.write(f"com_per_episode0:{total_0_com/eval_episodes}")
            f.write(f"com_per_episode1:{total_1_com/eval_episodes}")
            f.write(f"character_per_episode0:{total_0_charaters/eval_episodes}")
            f.write(f"charactor_per_episode1:{total_1_charaters/eval_episodes}")


        self.logger.info(f"eval done, avg transport rate {avg_finish}")
        self.logger.info("time: {}".format(time.time() - start))
        return avg_finish

    def close(self):
        """
        关闭环境，释放资源
        """
        self.env.close()


def init_logs(output_dir, name="simple_example"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler() # 控制台输出
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 新增一个logger用于记录时间信息
    time_logger = logging.getLogger(f"{name}_time")
    time_logger.setLevel(logging.DEBUG)
    time_fh = logging.FileHandler(os.path.join(output_dir, "time.log"))
    time_fh.setLevel(logging.DEBUG)
    time_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    time_fh.setFormatter(time_formatter)
    time_logger.addHandler(time_fh)

    return logger, time_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type=str, default="try")
    parser.add_argument("--run_id", type=str, default="run_0")
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="dataset/dataset_train/")
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs="+", type=str, default=("h_agent",))
    parser.add_argument(
        "--eval_episodes",
        nargs="+",
        default=(-1,),
        type=int,
        help="which episodes to evaluate on",
    )
    parser.add_argument(
        "--max_frames", default=3000, type=int, help="max frames per episode"
    )
    parser.add_argument("--no_launch_build", action="store_true")
    parser.add_argument("--communication", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_gt_mask", action="store_true")
    # LLM parameters
    parser.add_argument(
        "--source",
        default="deepseek",
        choices=["hf", "openai", "deepseek"],
        help="openai API or load huggingface models",
    )
    parser.add_argument(
        "--lm_id",
        default="gpt-3.5-turbo",
        help="name for openai engine or huggingface model name/path",
    )
    parser.add_argument(
        "--prompt_template_path",
        default="LLM/prompt_single.csv",
        help="path to prompt template file",
    )
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument(
        "--cot", action="store_true", help="use chain-of-thought prompt"
    )
    parser.add_argument(
        "--echo", action="store_true", help="to include prompt in the outputs"
    )
    parser.add_argument("--screen_size", default=512, type=int)
    parser.add_argument(
        "--no_save_img", action="store_true", help="do not save images", default=False
    )
    args = parser.parse_args()

    args.number_of_agents = len(args.agents)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok=True)
    logger,time_logger = init_logs(args.output_dir)

    challenge = Challenge_oppo(
        logger,
        time_logger,
        args.port,
        args.data_path,
        args.output_dir,
        args.number_of_agents,
        args.max_frames,
        not args.no_launch_build,
        screen_size=args.screen_size,
        data_prefix=args.data_prefix,
        gt_mask=not args.no_gt_mask,
        save_img=not args.no_save_img,
    )
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == "h_agent":
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == "lm_agent_oppo":
            agents.append(
                lm_agent_oppo(i, logger, args.max_frames, args, args.output_dir)
            )
        elif agent == "lm_agent":
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))
        else:
            pass
    try:
        challenge.submit(agents, logger, args.eval_episodes)
        # 提交进入chanllenge遍历执行
    finally:
        challenge.close()


if __name__ == "__main__":
    main()
