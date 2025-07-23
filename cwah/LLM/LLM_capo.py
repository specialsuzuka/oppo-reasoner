import random
import openai
import torch
import json
import os
import pandas as pd
from openai import OpenAIError
import backoff
from openai import OpenAI
import logging
from datetime import datetime

from LLM import LLM

class LLM_capo(LLM):
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
				 lm_id,
				 prompt_template_path,
				 communication,
				 cot,
				 sampling_parameters,
				 agent_id
                 ):
        super().__init__(source,  # 'huggingface' or 'openai'
                    lm_id,
                    prompt_template_path,
                    communication,
                    cot,
                    sampling_parameters,
                    agent_id)
        self.prompt_template_path = prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.host = 2-self.agent_id

        #communication counting:
        self.tokens = 0
        self.api = 0


        if self.host:
            self.meta_plan_init_prompt = df["prompt"][0]
            self.host_prompt = df["prompt"][1]
            self.temmate_prompt = None
            self.refiner_prompt = df["prompt"][3]
            self.parsing_prompt = df["prompt"][4]
        else:
            self.meta_plan_init_prompt = None
            self.host_prompt = None
            self.temmate_prompt = df["prompt"][2]
            self.refiner_prompt = None
            self.parsing_prompt = df["prompt"][4]

        if self.source == 'openai':
            client = OpenAI(
                api_key="sk-tkQC6suw159dxQoCkSrf2pTmSbIBawo7pP15FQN7d5vfTCxO",
                base_url="https://api.agicto.cn/v1"
            )
            print(f"loading openai model =============={lm_id}")
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif source == 'huggingface':
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                # 'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'early_stopping': True,
            }
        elif source == "debug":
            self.sampling_params = sampling_parameters
        else:

            raise ValueError("invalid source")
        

        def lm_engine (source, lm_id, device):
            if source == 'huggingface':
                from transformers import AutoModelForCausalLM, AutoTokenizer, LLaMATokenizer, LLaMAForCausalLM
                print(f"loading huggingface model {lm_id}")
                if 'llama' in lm_id or 'alpaca' in lm_id:
                    tokenizer = LLaMATokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce') # '/gpfs/u/scratch/AICD/AICDhnng/.cache')
                    model = LLaMAForCausalLM.from_pretrained(lm_id, # device_map="balanced_low_0",
                                                                # max_memory = {0: "10GB", 1: "20GB", 2: "20GB", 3: "20GB",4: "20GB",5: "20GB",6: "20GB",7: "20GB"},
                                                                torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                                load_in_8bit=False,
                                                                cache_dir='/work/pi_chuangg_umass_edu/.cahce')\
                                                                .to(device)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(lm_id, cache_dir='/work/pi_chuangg_umass_edu/.cahce')
                    model = AutoModelForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16,
                                                                    pad_token_id=tokenizer.eos_token_id,
                                                                    cache_dir='/work/pi_chuangg_umass_edu/.cahce').to(
                        device)
                print(f"loaded huggingface model {lm_id}")
            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            response = client.chat.completions.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            self.tokens += response.usage.completion_tokens
                            self.api += 1
                            if self.debug:
                                with open(f"LLM/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response.to_dict(), indent=4))
                                    f.write('\n')
                            generated_samples = [
                                choice.message.content 
                                for choice in response.choices 
                            ]
                            if 'gpt-4' in self.lm_id:
                                usage_cost = (response.usage.prompt_tokens * 0.03 / 1000 + 
                                        response.usage.completion_tokens * 0.06 / 1000)
                            elif 'gpt-3.5' in self.lm_id:
                                usage = ( response.usage.prompt_tokens + response.usage.completion_tokens ) * 0.002 / 1000
                            elif 'deepseek-r1' in self.lm_id:
                                usage = (( response.usage.prompt_tokens * 0.0024 )+ (response.usage.completion_tokens * 0.0096)) / 1000
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            if self.debug:
                                with open(f"LLM/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [
                                choice.message.content 
                                for choice in response.choices  # 直接遍历 choices 对象
                            ]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'huggingface':
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
        
                    output_dict = model.generate(input_ids, 
                                                **sampling_params)
                    generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                    for i, sample in enumerate(generated_samples):
                        stop_idx = sample.index('\n') if '\n' in sample else None
                        generated_samples[i] = sample[:stop_idx]
                elif source == "debug":
                    return ["navigation"]
                else:
                    raise ValueError("invalid source")
                if self.debug:
                    print(f"generated_samples: {generated_samples}")
                return generated_samples, usage

            return _generate
        self.generator = self.generator = lm_engine(self.source, self.lm_id, self.device)
        
    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored):

        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        """
        available_plans = []
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans
    
    def meta_plan_init(self):
        prompt = self.meta_plan_init_prompt
        prompt = prompt.replace("$GOAL$",self.goal_desc)
        chat_prompt = [{"role": "user", "content": prompt}]
        output,usage = self.generator(chat_prompt,self.sampling_params)
        meta_plan = output[0]
    
        return meta_plan
    def disscuss_refine(self,refine,meta_plan,oppo_progress,current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
        if self.host:
            if refine:
                prompt = self.refiner_prompt
            else:
                prompt = self.host_prompt
        else:
            prompt = self.temmate_prompt
        prompt = prompt.replace("$GOAL$",self.goal_desc)
        prompt = prompt.replace("$PREVIOUS\_PLAN$",meta_plan)#meta plan need something to send
        prompt = prompt.replace("$DIALOGUE\_HISTORY$",dialogue_history_desc)
        prompt = prompt.replace("$PROGRESS$",progress_desc)
        prompt = prompt.replace("$OPP\_PROGRESS$",oppo_progress)
        chat_prompt = [{"role": "user", "content": prompt}]
        output,usage = self.generator(chat_prompt,self.sampling_params)
        message = output[0]
        return message
    def parsing(self, meta_plan,current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored = None):
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
        prompt = self.parsing_prompt
        prompt = prompt.replace("$GOAL$",self.goal_desc)
        prompt = prompt.replace("$META\_PLAN$",meta_plan)
        prompt = prompt.replace("$DIALOGUE\_HISTORY$",dialogue_history_desc)
        prompt = prompt.replace("$PROGRESS$",progress_desc)
        prompt = prompt.replace("$PREVIOUS\_ACTIONS$",action_history_desc)
        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects,unchecked_containers,ungrabbed_objects,room_explored)
        prompt = prompt.replace("$ACTION\_LIST$",available_plans)
        
        chat_prompt = [{"role": "user", "content": prompt}]
        output,usage = self.generator(chat_prompt,self.sampling_params)
        output = output[0]
        plan = self.parse_answer(available_plans_list, output)
        return plan

    def progress_sending(self,current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored=None):
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        return progress_desc
    



