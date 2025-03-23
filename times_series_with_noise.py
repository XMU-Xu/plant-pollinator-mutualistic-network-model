# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:59:43 2025

@author: XUFEI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def box_muller():
    u1 = np.random.uniform(0, 1)
    u2 = np.random.uniform(0, 1)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0

# 定义洛伦兹系统的随机微分方程
def sss_sde(t, state, noise_strength):
    p, a = state
    # 确定性部分
    dp = alpha_p * p - beta * p**2 + gamma_p * a * p / (1 + h * gamma_p * a) + mu
    da = alpha_a * a - beta * a**2 + gamma_a * p * a / (1 + h * gamma_a * p) + mu - kappa * a
    
    # 随机噪声部分
    noise_p = noise_strength * box_muller()
    noise_a = noise_strength * box_muller()
    
    
    # 加入随机噪声
    dp += noise_p
    da += noise_a
    
    return [dp, da]

# 定义求解SDE的函数
def solve_sde_sss(t_span, initial_state, dt, noise_strength):
    num_steps = int((t_span[1] - t_span[0]) / dt)
    states = np.zeros((num_steps, 2))
    times = np.linspace(t_span[0], t_span[1], num_steps)
    
    # 初始条件
    states[0] = initial_state
    
    # 通过Euler-Maruyama方法迭代求解
    for i in range(1, num_steps):
        dt_actual = times[i] - times[i - 1]
        states[i] = states[i - 1] + np.array(sss_sde(times[i - 1], states[i - 1], noise_strength)) * dt_actual
        states[i] = [x if x >= 0 else 0 for x in states[i]]
    
    return times, states

if __name__ == "__main__":
    
    alpha_p = 0.1
    alpha_a = 0.1
    
    beta = 1.0
    
    h = 0.4
    
    mu = 0.0001
    
    kappa = 0.70
    
    gamma_p = 3.349081
    gamma_a = 1.902356

    # 定义SDE中的噪声强度
    noise_strength = 0.5
    
    # 设置初始条件和时间范围
    initial_state =  [0.10178, 2.37792e-4]#[1.21562, 0.60148]
    #initial_state = [1.1678, 0.55647] #[0.10174, 2.26972e-4]
    #initial_state =  [1.10642, 0.50292]#[0.10171, 2.17095e-4]
    t_span = (0, 10**4)
    dt = 0.01
    
    # 求解随机3S系统
    times, states = solve_sde_sss(t_span, initial_state, dt, noise_strength)
    
    # 绘图展示结果
    fig = plt.figure(figsize=(12, 6))
    
    # 2D相图
    ax1 = fig.add_subplot(131)
    ax1.plot(states[:,0], states[:,1], lw=0.5)
    ax1.set_title("Stochastic PA System (pa Plane)")
    ax1.set_xlabel('u')
    ax1.set_ylabel('w')
    
    # 时间序列图
    ax2 = fig.add_subplot(132)
    ax2.plot(times, states[:,0], lw=0.5)
    ax2.set_title("Stochastic PA System (Time Series)")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('p')
    
    # 时间序列图
    ax3 = fig.add_subplot(133)
    ax3.plot(times, states[:,1], lw=0.5)
    ax3.set_title("Stochastic PA System (Time Series)")
    ax3.set_xlabel('Time')
    ax3.set_ylabel('a')
    
    plt.tight_layout()
    plt.show()
    
    # 创建 DataFrame  
    data = {  
        't': times,  
        'y0': states[:,0],  
        'y1': states[:,1] 
    }  
    df = pd.DataFrame(data)  
      
    # 将 DataFrame 保存为 CSV 文件  
    df.to_csv('kappa=' + str(kappa) + '_solution1.csv', index=False)