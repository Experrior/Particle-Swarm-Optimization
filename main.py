# This is a sample Python script.
import copy
import os
import random
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from math import log10, floor


def round_to_first_significant_digit(number):
    return round(number, -int(floor(log10(abs(number)))))

def generate_nd_function(seed: int, num_of_functions: int, n_variables: int, coefficients_range: [2]):
    random.seed(seed)
    coef_low = coefficients_range[0]
    coef_high = coefficients_range[1]

    functions_pool = [np.sin, np.cos]

    functions_default = []
    for _ in range(num_of_functions):
        functions_default.append(functions_pool[random.randint(0, len(functions_pool) - 1)])

    # def order of variables
    variables_default = [x for x in range(min(num_of_functions, n_variables))]
    for _ in range(min(num_of_functions, n_variables), num_of_functions):
        variables_default.append(random.randint(0, n_variables - 1))

    # def coefficients
    coefficients_default = []
    for _ in range(num_of_functions * 2):
        if random.random() > 0.5:
            coefficients_default.append(coef_low + (random.gauss(0, 0.3) * (coef_high - coef_low)))
        else:
            coefficients_default.append(coef_low + (random.random() * (coef_high - coef_low)))

    def function(input_values, functions=functions_default, coefficients=coefficients_default, seed=seed,
                 variables=variables_default):
        random.seed(seed)
        result = 0
        for i in range(len(functions)):
            if random.random() < 0.33:
                result += coefficients[i * 2] * functions[i](coefficients[i * 2 + 1] * input_values[variables[i]])
            elif _ < 0.66:
                result += coefficients[i * 2] * functions[i](coefficients[i * 2 + 1] * input_values[0])
            else:
                result += coefficients[i * 2] * functions[i](coefficients[i * 2 + 1] * input_values[-1])

        return result

    return function


def complex_2d_function(xy: []):
    x = xy[0]
    y = xy[1]
    return (
            np.sin(9.7 * x) + np.exp(np.sin(7.3 * x)) - np.cos(2 * np.sin(4 * x ** 2)) +
            np.sin(0.4 * y) + np.exp(np.cos(11.3 * y)) - np.cos(7 * np.sin(y ** 2)) -
            np.sin(1.3 * x) + np.cos(2 * y) - np.cos(x * y ** 2) ** 2 +
            np.exp(np.sin(np.exp(np.cos(x ** 2)))) - np.exp(np.cos(np.exp(np.cos(y ** 2)))) +
            np.sin(3 * x * y) - np.cos(x / (1 + (7 * abs(y)))) + np.sin((5 * y) / 1 + abs(x)) -
            2 * np.sin(1 / 1 + x ** 2) + np.cos(1 - x) + np.sin(-10 * y + 2) +
            np.sin(2 * x) + np.sin(12 * x) - np.sin(17 * x) - np.cos(15 * y) -
            np.cos(2 * y) + np.cos(15 * y)
    )


def limit(value, lower_limit, upper_limit):
    return max(lower_limit, min(value, upper_limit))


def solve_by_pso(function, num_of_vars, init_particles, low_boundary, high_boundary, speed, loop, reset, decay,
                 exploitation, exploration, randomize=False, neighbourhood=False, get_metadata=False):


    speed_limit = (abs(high_boundary - low_boundary)) / 25

    # init particles positions/speed
    particles = []
    for i in range(init_particles):
        particles.append(
            [[low_boundary + random.Random().random() * (high_boundary - low_boundary) for _ in range(num_of_vars)],
             [(low_boundary + random.Random().random() * (high_boundary - low_boundary)) * speed for _ in
              range(num_of_vars)]])
    if get_metadata:
        particles = np.array(particles)

    # get best value/pos
    personal_best = [[particles[0][0], function(particles[0][0])]]
    best_position = particles[0][0]
    best_value = function(best_position)
    prev_best = best_value
    for particle in particles[1:]:
        if function(particle[0]) < best_value:
            best_position = particle[0]
            best_value = function(best_position)
        personal_best.append([particle[0], function(particle[0])])

    if get_metadata:
        list_of_particle_states = [list(particles[:, 0, :])]
        list_of_best_positions = [best_position]
    for i in range(loop):

        # update speed, and move particle
        count = 0
        if randomize:
            for particle in particles:
                particle[1] = [limit((particle[1][x] * (1 - decay)) + (
                        random.random() * exploitation * (best_position[x] - particle[0][x]))
                                     + (random.random() * exploration * (personal_best[count][1] - particle[0][x])),
                                     -speed_limit, speed_limit) for x
                               in range(num_of_vars)]

                particle[0] = [max(low_boundary, min(a + b, high_boundary)) for a, b in zip(particle[0], particle[1])]

                count += 1
        else:
            for particle in particles:
                # print("speed before", particle[1],i)
                particle[1] = [limit((particle[1][x] * (1 - decay)) + (
                        1 * exploitation * (best_position[x] - particle[0][x]))
                                     + (1 * exploration * (personal_best[count][1] - particle[0][x])), -speed_limit,
                                     speed_limit) for x
                               in range(num_of_vars)]
                # print("pos,speed:",particle[0], particle[1],i)
                particle[0] = [limit(a + b, low_boundary, high_boundary) for a, b in zip(particle[0], particle[1])]
                # print("pos after", particle[0],i)
                count += 1

        # update personal best
        for x in range(init_particles):
            if function(particles[x][0]) < personal_best[x][1]:
                personal_best[x][0] = particles[x][0]
                personal_best[x][1] = function(particles[x][0])

        # update best positions
        for particle in particles:
            if function(particle[0]) < best_value:
                best_position = copy.deepcopy(particle[0])
                best_value = function(best_position)

        if get_metadata:
            list_of_particle_states.append(copy.deepcopy(list(particles[:, 0, :])))
            list_of_best_positions.append(copy.deepcopy(best_position))

        # print(prev_best, best_value, abs((best_value - prev_best) / prev_best))
        # if abs((best_value - prev_best) / prev_best) < 0.0001:
        #     count += 1
        # else:
        #     count = 0
        # prev_best = best_value
        # if count > 10:
        #     break

        #     reset particles
        for particle in particles:
            if random.random() < reset:
                particle[0] = [low_boundary + random.Random().random() * (high_boundary - low_boundary) for _ in
                               range(num_of_vars)]
                particle[1] = [(low_boundary + random.Random().random() * (high_boundary - low_boundary)) * speed for _
                               in
                               range(num_of_vars)]
    print("finished PSO")
    if get_metadata:
        name_1 = "seed=" + str(seed) + ", speed=" + str(speed) + ", decay=" + str(decay) +\
                 ", randomize=" + str(randomize)
        name_2 = "exploitation=" + str(exploitation) + ", exploration=" + str(exploration)
        return best_position, best_value, list_of_particle_states, list_of_best_positions, name_1, name_2
    else:
        return best_position, best_value

def visualize_pso_2d(function, list_of_particle_states, list_of_best_positions, name_1, name_2, verbose=True, round=True):
    if not os.path.exists("pngs"):
        os.mkdir("pngs")

    num_of_frames = len(list_of_best_positions)
    name_1 = "seed=" + str(seed) + ", speed=" + str(round_to_first_significant_digit(speed)) +\
             ", decay=" + str(round_to_first_significant_digit(decay))
    name_2 = "exploitation=" + str(round_to_first_significant_digit(exploitation)) +\
             ", exploration=" + str(round_to_first_significant_digit(exploration))

    def plot(k, particles, best_particle):

        num = 100
        # Example evaluation
        x = np.linspace(low_boundary, high_boundary, num)
        y = np.linspace(low_boundary, high_boundary, num)
        X, Y = np.meshgrid(x, y)
        Z = [[]]
        for ex in range(num):
            for i in range(num):
                Z[-1].append(function([X[ex][i], Y[ex][i]]))
            Z.append([])
        Z.pop(num)
        Z = np.array(Z)
        # Visualize

        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Function Value')
        plt.title("Best value: " + str(function(best_particle)) + "\n" + name_1 + "\n" + name_2, fontsize=7)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(x=np.array(particles)[:, 0], y=np.array(particles)[:, 1], c="red", s=0.2)
        plt.scatter(x=[best_particle[0]], y=best_particle[1], s=5.4, c="white")
        plt.savefig("pngs/" + str(k) + ".png", dpi=300)
        if k == 0:
            plt.show()
        plt.close()

    if verbose:
        print("Started generating images")
    for i in range(num_of_frames):
        plot(i, list_of_particle_states[i], list_of_best_positions[i])
        if verbose:
            print(100 * (i + 1) / num_of_frames, "%")

    if os.path.exists(name_1 + ", " + name_2 + ".gif"):
        os.remove(name_1 + ", " + name_2 + ".gif")

    # Generating the gif with all the frames:
    frames = np.stack([iio.imread(f"pngs/{x}.png") for x in range(num_of_frames)], axis=0)
    # saving the gif
    iio.imwrite(name_1 + ", " + name_2 + ".gif", frames, loop=4, duration=1)

    if os.path.exists(name_1 + ", " + name_2 + ".gif"):
        print("Successfully saved gif: \n" + name_1 + ", " + name_2 + ".gif")
    else:
        print("Something went wrong...")



if __name__ == '__main__':
    seed = 66
    n_functions = 30
    num_of_variables = 2
    variable_ranges = [-2, 2]
    function_2d = generate_nd_function(seed, n_functions, num_of_variables, variable_ranges)
    # function_2d = complex_2d_function
    low_boundary = - 3
    high_boundary = 3
    loop = 200
    reset = 0.0
    particles = 20
    speed = 0.1 ** 3
    decay = 0.1
    exploration = 0.1 ** 3
    exploitation = 0.01
    neighbourhood = False
    randomize = True

    result_list = solve_by_pso(function_2d, num_of_variables,
                               init_particles=particles, low_boundary=low_boundary, high_boundary=high_boundary,
                               loop=loop, speed=speed, reset=reset, decay=decay, exploration=exploration,
                               exploitation=exploitation, neighbourhood=neighbourhood, randomize=randomize,
                               get_metadata=True)

    visualize_pso_2d(function_2d, *result_list[2:], verbose=True)

    exit()
