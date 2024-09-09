import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Car parameters
m = 1.0  # mass of the car
torque = 1.0  # constant torque
max_steering_angle = np.pi / 4  # max steering angle (45 degrees)

# Environment (obstacles and goal)
goal_position = np.array([10.0, 10.0])
obstacle_positions = [np.array([5.0, 5.0]), np.array([7.0, 8.0])]
obstacle_radius = 1.0

# Parameters for reward system
alpha = 10.0  # penalty for being far from the line
beta = 50.0  # penalty for being near an obstacle
safe_distance = 1.0  # distance to consider for obstacle penalty

def compute_reward(state, obstacle_positions, line_direction, start_position, goal_position):
    x, y = state[0], state[1]
    # Distance from the line
    point = np.array([x, y])
    projection = start_position + np.dot(point - start_position, line_direction) * line_direction
    distance_from_line = np.linalg.norm(point - projection)

    # Obstacle penalties
    obstacle_penalty = 0.0
    for obs in obstacle_positions:
        dist_to_obs = np.linalg.norm(point - obs)
        if dist_to_obs < safe_distance:
            obstacle_penalty += (safe_distance - dist_to_obs)  # penalty for being too close

    # Compute total penalty
    total_penalty = alpha * distance_from_line + beta * obstacle_penalty

    # Reward is negative of the penalty
    return -total_penalty
# Define the car's dynamics using continuous functions (no if-else)
def car_dynamics(state, t, torque, goal_position, obstacle_positions):
    x, y, v, theta = state  # state = [x, y, velocity, orientation]

    # Linear acceleration due to torque
    acceleration = torque / m

    # Steering angle is continuously changing to minimize the goal potential
    dx_goal, dy_goal = goal_position - np.array([x, y])
    desired_angle = np.arctan2(dy_goal, dx_goal)
    steering_angle = np.clip(desired_angle - theta, -max_steering_angle, max_steering_angle)

    # Velocity dynamics
    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dvdt = acceleration
    dthetadt = steering_angle  # simple proportional steering control

    # Obstacle avoidance potential fields
    for obs in obstacle_positions:
        dx_obs, dy_obs = x - obs[0], y - obs[1]
        dist_to_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2)
        if dist_to_obs < obstacle_radius:
            # Add a repulsive potential force away from the obstacle
            avoid_angle = np.arctan2(dy_obs, dx_obs)
            steering_angle += np.clip(avoid_angle - theta, -max_steering_angle, max_steering_angle)

    return [dxdt, dydt, dvdt, dthetadt]


# Time settings
t = np.linspace(0, 10, 500)  # simulate for 10 seconds

# Initial conditions [x, y, v, theta]
initial_state = [0.0, 0.0, 0.0, 0.0]

# Simulate the car's motion using odeint
result = odeint(car_dynamics, initial_state, t, args=(torque, goal_position, obstacle_positions))

# Extract the positions for plotting
x_vals, y_vals = result[:, 0], result[:, 1]

# Plot the car's trajectory, goal, and obstacles
plt.figure(figsize=(8, 8))
plt.plot(x_vals, y_vals, label="Car Path")
plt.scatter(*goal_position, color='green', label="Goal", s=100)

# Plot the obstacles
for obs in obstacle_positions:
    circle = plt.Circle(obs, obstacle_radius, color='red', alpha=0.5)
    plt.gca().add_patch(circle)
plt.scatter(*zip(*obstacle_positions), color='red', label="Obstacles", s=100)

plt.title("Car Path Simulation with Differentiable Control")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
