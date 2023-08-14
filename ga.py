import numpy as np
from copy import deepcopy

def genetic_algorithm(problem, **kwargs):
    """
    遗传算法用于路径规划。

    参数:
        problem (dict): 包含成本函数和变量信息的问题定义。
        max_iter (int): 算法的最大迭代次数。默认值为100。
        pop_size (int): 种群规模。默认值为100。
        mutation_rate (float): 变异率。默认值为0.01。
        crossover_rate (float): 交叉率。默认值为0.8。
        callback (callable): 在每次迭代后调用的回调函数。默认值为None。

    返回:
        dict: 找到的最佳解和种群历史记录。
    """

    max_iter = kwargs.get('max_iter', 100)
    pop_size = kwargs.get('pop_size', 100)
    mutation_rate = kwargs.get('mutation_rate', 0.05)
    crossover_rate = kwargs.get('crossover_rate', 0.8)
    callback = kwargs.get('callback', None)
    _last_cost = 0
    # 空个体模板
    empty_individual = {
        'position': None,
        'cost': None,
        'details': None,
    }

    # 提取问题信息
    cost_function = problem['cost_function']
    var_min = problem['var_min']
    var_max = problem['var_max']
    num_var = problem['num_var']

    # 初始化种群
    population = []
    for _ in range(pop_size):
        individual = deepcopy(empty_individual)
        individual['position'] = np.random.uniform(var_min, var_max, num_var)
        individual['cost'], individual['details'] = cost_function(individual['position'])
        population.append(individual)

    # 按成本升序对种群进行排序
    population.sort(key=lambda x: x['cost'])

    # 最佳和最差个体
    best_individual = population[0]
    worst_individual = population[-1]

    # 种群历史记录
    pop_history = []
    pop_history.append(deepcopy(population))

    # 遗传算法循环
    for it in range(max_iter):
        if it > max_iter//2:
            mutation_rate = 0.5
        # 创建一个空的新一代种群
        new_population = []

        # 进行精英主义：在新种群中保留最佳个体
        new_population.append(deepcopy(best_individual))

        # 交叉和变异
        for _ in range(1, pop_size):
            parent1 = select_parent(population)
            parent2 = select_parent(population)

            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)

            child['cost'], child['details'] = cost_function(child['position'])

            new_population.append(child)

        # 用新种群替换旧种群
        population = new_population
        counter = 0
        if counter == 0:
            print( population[0] )
            counter+=1
        # 按成本升序对种群进行排序
        population = sorted(population,key=lambda x: x['cost'])

        # 更新最佳和最差个体
        best_individual = population[0]
        worst_individual = population[-1]

        # 将当前种群添加到历史记录
        pop_history.append(deepcopy(population))

        if callable(callback):
             callback({
                'it': it + 1,
                'best_individual': best_individual,
                'population': population,
            })
        # print(f'迭代 {it + 1}: 最佳成本 = {best_individual["cost"]}')
        if it+1 % 10 == 0:
            length = best_individual['cost']
            if (_last_cost == length):
                 break
            _last_cost = length
    return best_individual, pop_history


def select_parent(population):
    """
    使用锦标赛选择从种群中选择一个父代。

    参数:
        population (list): 个体的种群。

    返回:
        dict: 选定的父代个体。
    """

    tournament_size = 3
    tournament = np.random.choice(population, size=tournament_size, replace=False)
    return sorted(tournament, key=lambda x: x['cost'])[0]


def crossover(parent1, parent2, crossover_rate):
    """
    在两个父代之间执行交叉，产生一个子代。

    参数:
        parent1 (dict): 第一个父代个体。
        parent2 (dict): 第二个父代个体。
        crossover_rate (float): 交叉率。

    返回:
        dict: 创建的子代个体。
    """

    child = deepcopy(parent1)

    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1['position']))
        child['position'][:crossover_point] = parent2['position'][:crossover_point]

    return child


def mutate(individual, mutation_rate):
    """
    对给定的个体进行变异。

    参数:
        individual (dict): 要进行变异的个体。
        mutation_rate (float): 变异率。

    返回:
        dict: 变异后的个体。
    """

    for i in range(len(individual['position'])):
        if np.random.rand() < mutation_rate:
            individual['position'][i] = np.random.uniform(individual['position'][i] - 1, individual['position'][i] + 1)

    return individual
