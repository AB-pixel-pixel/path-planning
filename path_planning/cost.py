from path_planning.solution import SplinePath
from path_planning.environment import Environment
import numpy as np

from scipy.interpolate import CubicSpline

def curvature(path):
    """
    计算路径的曲率。

    参数:
        path (numpy.ndarray): 输入路径。

    返回:
        numpy.ndarray: 路径的曲率。
    """

    dx_dt = np.gradient(path[:, 0])
    dy_dt = np.gradient(path[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    denominator = dx_dt ** 2 + dy_dt ** 2
    zero_mask = denominator == 0.0

    curvature = np.zeros_like(dx_dt)
    curvature[~zero_mask] = np.abs(dx_dt[~zero_mask] * d2y_dt2[~zero_mask] - dy_dt[~zero_mask] * d2x_dt2[~zero_mask]) / \
                            denominator[~zero_mask] ** (3 / 2)

    return curvature


def smoothness_penalty(path, threshold):
    """
    计算路径的圆滑度损失。

    参数:
        path (numpy.ndarray): 输入路径。
        threshold (float): 圆滑度的阈值。超过该阈值将被认为是不够圆滑。

    返回:
        float: 圆滑度损失。
    """

    curvatures = curvature(path)
    roughness = np.sum(curvatures[curvatures > threshold])
    return roughness


START_VIOLATION_PENALTY = 1
GOAL_VIOLATION_PENALTY = 1
ENV_VIOLATION_PENALTY = 10000
COLLISION_PENALTY = 10000
ACKERMANN_CONSTRAINT_PENALTY = 5  # Penalty for violating the Ackermann steering constraint
THRESHOLD = 5


def PathPlanningCost(sol: SplinePath):

    # Get path
    path = sol.get_path()

    # Length of path
    length = sol.environment.path_length(path)

    # Violations of path
    _, details = sol.environment.count_violations(path)

    # Cost
    cost = length

    # Add penalty for start violation
    if details['start_violation']:
        cost *= 1 + START_VIOLATION_PENALTY

    # Add penalty for goal violation
    if details['goal_violation']:
        cost *= 1 + GOAL_VIOLATION_PENALTY

    # Environment violation
    if details['environment_violation']:
        cost *= 1 + details['environment_violation_count']*ENV_VIOLATION_PENALTY + ENV_VIOLATION_PENALTY

    # Collision violation
    if details['collision_violation']:
        cost *= 1 + details['collision_violation_count']*COLLISION_PENALTY + COLLISION_PENALTY


    # Check Ackermann steering constraint violation
    cost  *= 1 + smoothness_penalty(path,THRESHOLD)


    # Add details
    details['sol'] = sol
    details['path'] = path
    details['length'] = length
    details['cost'] = cost
    
    return cost, details

def EnvCostFunction(environment: Environment, resolution=100):
    def CostFunction(xy):
        sol = SplinePath.from_list(environment, xy, resolution, normalized=True)
        return PathPlanningCost(sol)
    
    return CostFunction
