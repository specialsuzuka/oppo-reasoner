指标需要做的：
- 写一个API调用计数 shaokang
- 写一个通信字数计数 shaokang
    - 包括字符数，单词数，可以从dialogue_history前后交互的地方找变量计算 
- 想一想冗余行为怎么计算 TODO


观测实验方面：
- 对话内容记录 zhimin
    - 包含对话文本；任务进程；各自的action history；各自的obs；最好有各自视角的图片；
- 会不会出现通信冲突的问题 CoELA + capo


代码基本框架框架：
- challenge_oppo.py 所有算法公用
- LLM/prompts
    - capo prompt 从论文copy保存为csv shaoakng
    - oppo-reasoner prompt zhimin 包含planning active 和 passive两种
- LM_agent_* 一种算法一个 负责维护最核心的算法 各个module在这里组织
    - LM_agent_oppo_v2.py 与方法对齐 TODO zhimin lead
        - reasoning-enhanced meomery module TODO
    - LM_agent_capo.py shaokang
- LLM_*.py 一种算法一个
    - LLM_capo.py shaokang
    - LLM_oppo_v2.py zhimin lead
        - reasoning-guided planning module TODO
        - reasoning-triggered collaboration module TODO
 
补充Roco reading shaokang
- frame 模拟器的步进 top-down img 单位frame
- step low_action: move(0,5m) take 5 frames turn left turn right img 第一人称的
- planning_step : LLM output gotoroom() 重点 决策步

go to room(living room<4004>) -> lm_agent.py go to room -> move(target pos) return action -> env

capo : discussion 3 steps
1. go to livingroom
2. 模糊
3. 模糊