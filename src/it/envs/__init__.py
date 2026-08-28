"""Isaac Lab 任务环境。

`plan/04` §2：**两个执行器，两个实验**。本模块提供的是环境（物理 + reward +
观测），Expert / E-T / E-I 三者共用同一套环境，只换 reward 与观测字段。

S2 阶段只用 Privileged Expert 的配置：观测给上帝视角，reward 给任务结果。
"""
