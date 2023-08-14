import path_planning as pp
import matplotlib.pyplot as plt
from pso import PSO

plt.rcParams["figure.autolayout"] = True

# Create environment
# env_params = {
#     'width': 100,
#     'height': 100,
#     'robot_radius': 1,
#     'start': [5,5],
#     'goal': [95,95],
# }
# env = pp.Environment(**env_params)
#
# # Obstacles
# obstacles = [
#     {'center': [20, 40], 'radius': 5},
#     {'center': [30, 30], 'radius': 9},
#     {'center': [30, 70], 'radius': 10},
#     {'center': [50, 10], 'radius': 8},
#     {'center': [60, 80], 'radius': 15},
#     {'center': [70, 40], 'radius': 12},
#     {'center': [80, 20], 'radius': 7},
# ]
env_params = {
    'width': 100,
    'height': 100,
    'robot_radius': 1,
    'start': [5,5],
    'goal': [80,5],
}
env = pp.Environment(**env_params)

# Obstacles
obstacles = [
    {'center': [40, 20], 'radius': 20},
    {'center': [40, 60], 'radius': 20},
    {'center': [70, 60], 'radius': 10},
    {'center': [85, 60], 'radius': 5},
]
for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

# Create cost function
num_control_points = 3
resolution = 50
cost_function = pp.EnvCostFunction(env, resolution)

# Optimization Problem
problem = {
    'num_var': 2*num_control_points,
    'var_min': 0,
    'var_max': 1,
    'cost_function': cost_function,
}

# Callback function
path_line = None
def callback(data):
    global path_line
    it = data['it']
    sol = data['gbest']['details']['sol']
    if it==1 or it% 10 == 0:
        fig = plt.figure(figsize=[7, 7])
        pp.plot_environment(env)
        path_line = pp.plot_path(sol, color='b')
        plt.grid(True)
        length = data['gbest']['details']['length']
        plt.title(f"Iteration: {it}, Cost: {length:.2f}")
        plt.show(block=False)
    else:
        pp.update_path(sol, path_line)



# Run PSO
pso_params = {
    'max_iter': 100,
    'pop_size': 100,
    'c1': 2,
    'c2': 1,
    'w': 0.8,
    'wdamp': 1,
    'resetting': 25,
}
import time

# 开始计时
start_time = time.time()

# 执行你的代码
bestsol, pop = PSO(problem, callback=callback, **pso_params)
# 结束计时
end_time = time.time()

# 计算代码执行时间
execution_time = end_time - start_time

# 打印结果
print("代码执行时间：", execution_time, "秒")


plt.show()
