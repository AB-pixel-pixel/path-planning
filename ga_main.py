import path_planning as pp
import matplotlib.pyplot as plt
from ga import genetic_algorithm
plt.rcParams["figure.autolayout"] = True
#-----------------------------------------------------------------------------------------------------------------------
# Create environment


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

#-----------------------------------------------------------------------------------------------------------------------
for obs in obstacles:
    env.add_obstacle(pp.Obstacle(**obs))

# Create cost function
num_control_points = 5
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
last_cost = 0
def callback(data):
    global path_line
    it = data['it']
    sol = data['best_individual']['details']['sol']
    if it==1or it %10 == 0:
        fig = plt.figure(figsize=[7, 7])
        pp.plot_environment(env)
        path_line = pp.plot_path(sol, color='b')
        plt.grid(True)
        length = data['best_individual']['cost']
        plt.title(f"Iteration: {it}, Cost: {length:.2f}")
        plt.show(block=False)
    pp.update_path(sol, path_line)


# Run GA
ga_params = {
    'max_iter': 100,
    'pop_size': 100,
    'mutation_rate': 0.05,
    'crossover_rate': 0.8
}


import time

# 开始计时
start_time = time.time()

# 执行你的代码
bestsol, pop = genetic_algorithm(problem, callback=callback, **ga_params)
# 结束计时
end_time = time.time()

# 计算代码执行时间
execution_time = end_time - start_time

# 打印结果
print("代码执行时间：", execution_time, "秒")



plt.show()
