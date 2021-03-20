import argparse
import collections
import copy
import datetime
import math
import numpy as np
import os
import pickle
import random
import sys
import setproctitle
import tensorflow as tf
import scipy.io as scio
from tensorflow.contrib import rnn
from sequence_encoder import encode_seq, embed_seq
from path_encoder import p_encode_seq, p_embed_seq


os.environ["CUDA_VISIBLE_DEVICES"]="2"
setproctitle.setproctitle("l2i@zhouyikang")


EPSILON = 1e-6

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(args=None):
    parser = argparse.ArgumentParser(description="Meta optimization")
    parser.add_argument('--epoch_size', type=int, default=5120000, help='Epoch size')

    parser.add_argument('--num_lstm_units', type=int, default=128, help="number of LSTM units")
    parser.add_argument('--num_feedforward_units', type=int, default=128, help="number of feedforward units")
    parser.add_argument('--problem', default='vrp', help="the problem to be solved, {tsp, vrp}")
    parser.add_argument('--train_operators', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--depot_positioning', default='R', help="{R, C, E}")
    parser.add_argument('--customer_positioning', default='R', help="{R, C, RC}")

    parser.add_argument('--num_training_points', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_test_points', type=int, default=100, help="size of the problem for testing")
    parser.add_argument('--num_episode', type=int, default=40000, help="number of training episode")
    parser.add_argument('--max_num_rows', type=int, default=2000000, help="")
    parser.add_argument('--num_paths_to_ruin', type=int, default=2, help="")
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--max_rollout_steps', type=int, default=20000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=100000, help="maximum rollout time in seconds")
    parser.add_argument('--use_cyclic_rollout', type=str2bool, nargs='?', const=True, default=False, help="use cyclic rollout")
    parser.add_argument('--use_random_rollout',type=str2bool, nargs='?', const=True, default=False, help="use random rollout")
    parser.add_argument('--detect_negative_cycle', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--use_rl_loss', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--use_attention_embedding', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--epsilon_greedy', type=float, default=0.05, help="")
    parser.add_argument('--sample_actions_in_rollout', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--num_active_learning_iterations', type=int, default=1, help="")
    parser.add_argument('--max_no_improvement', type=int, default=6, help="")
    parser.add_argument('--debug_mode', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--debug_steps', type=int, default=1, help="")
    parser.add_argument('--num_actions', type=int, default=17, help="dimension of action space")
    parser.add_argument('--max_num_customers_to_shuffle', type=int, default=20, help="")
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    parser.add_argument('--input_embedded_trip_dim', type=int, default=15, help="")
    parser.add_argument('--input_embedded_trip_dim_2', type=int, default=15, help="")
    #parser.add_argument('--input_embedded_trip_dim_2', type=int, default=11, help="")
    parser.add_argument('--num_embedded_dim', type=int, default=64, help="")
    parser.add_argument('--num_embedded_dim_1', type=int, default=64, help="")
    parser.add_argument('--num_embedded_dim_2', type=int, default=64, help="dim")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")
    parser.add_argument('--use_history_action_distribution', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--step_interval', type=int, default=500)

    # './rollout_model_1850.ckpt'
    parser.add_argument('--model_to_restore', type=str, default=None, help="")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")

    parser.add_argument('--num_paths',type=int,default=5,help="")
    parser.add_argument('--max_path_distance',type=float,default=float('inf'), help="")
    parser.add_argument('--max_points_per_trip', type=int, default=50, help="upper bound of number of point in one trip")
    parser.add_argument('--max_trips_per_solution', type=int, default=15, help="upper bound of number of trip in one solution")
    parser.add_argument('--Velocity', type=float, default=60,help="Velocity")
    
    parser.add_argument('--one_vehicle_mode',type=str2bool,nargs='?',const=True,default=False,help="")
    parser.add_argument('--baseline_exp', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--with_test_set',type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--with_ranking',type=str2bool, nargs='?', const=True, default=False, help="")



    config = parser.parse_args(args)
    return config


config = get_config()
if config.max_no_improvement is None:
    config.max_no_improvement = config.num_actions


def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)


class Problem:
    def __init__(self, locations, capacities,TWs):
        self.locations = copy.deepcopy(locations)
        self.capacities = copy.deepcopy(capacities)
        self.TWs = copy.deepcopy(TWs)
        self.distance_matrix = []
        for from_index in range(len(self.locations)):
            distance_vector = []
            for to_index in range(len(self.locations)):
                distance_vector.append(calculate_distance(locations[from_index], locations[to_index]))
            self.distance_matrix.append(distance_vector)
        self.total_customer_capacities = 0
        for capacity in capacities[1:]:
            self.total_customer_capacities += capacity
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}
        self.num_solutions = 0
        self.num_traversed = np.zeros((len(locations), len(locations)))
        self.distance_hashes = set()

    def record_solution(self, solution, distance):
        self.num_solutions += 1.0 / distance
        for path in solution:
            if len(path) > 2:
                for to_index in range(1, len(path)):
                    #TODO: change is needed for asymmetric cases.
                    self.num_traversed[path[to_index - 1]][path[to_index]] += 1.0 / distance
                    self.num_traversed[path[to_index]][path[to_index - 1]] += 1.0 / distance
                    # for index_in_the_same_path in range(to_index + 1, len(path)):
                    #     self.num_traversed[path[index_in_the_same_path]][path[to_index]] += 1
                    #     self.num_traversed[path[to_index]][path[index_in_the_same_path]] += 1

    def add_distance_hash(self, distance_hash):
        self.distance_hashes.add(distance_hash)

    def get_location(self, index):
        return self.locations[index]

    def get_capacity(self, index):
        return self.capacities[index]

    def get_capacity_ratio(self):
        return self.total_customer_capacities / float(self.get_capacity(0))

    def get_num_customers(self):
        return len(self.locations) - 1

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]

    def get_frequency(self, from_index, to_index):
        return self.num_traversed[from_index][to_index] / (1.0 + self.num_solutions)

    def get_TW(self, index, StartorEnd):
        return self.TWs[index][StartorEnd]

    def reset_change_at_and_no_improvement_at(self):
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}

    def mark_change_at(self, step, path_indices):
        for path_index in path_indices:
            self.change_at[path_index] = step

    def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        self.no_improvement_at[key] = step

    def should_try(self, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        no_improvement_at = self.no_improvement_at.get(key, -1)
        return self.change_at[index_first] >= no_improvement_at or \
               self.change_at[index_second] >= no_improvement_at or \
               self.change_at[index_third] >= no_improvement_at


def calculate_distance_between_indices(problem, from_index, to_index):
    return problem.get_distance(from_index, to_index)


def calculate_adjusted_distance_between_indices(problem, from_index, to_index):
    distance = problem.get_distance(from_index, to_index)
    frequency = problem.get_frequency(from_index, to_index)
    # return (1.0 - frequency)
    return distance * (1.0 - frequency)
    # return distance * frequency


def calculate_trip_distance(trip):
    sum = 0.0
    for i in range(len(trip)):
        sum += calculate_distance(trip[i - 1], trip[i])
    return sum


def calculate_path_distance(problem, path):
    sum = 0.0
    for i in range(1, len(path)):
        sum += calculate_distance_between_indices(problem, path[i - 1], path[i])
    sum+=calculate_time_cost(problem,path)
    return sum


def calculate_solution_distance(problem, solution):
    total_distance = 0.0
    for path in solution:
        total_distance += calculate_path_distance(problem, path)
    return total_distance



def calculate_path_distance_dep(problem, path):
    sum = 0.0
    for i in range(1, len(path)):
        sum += calculate_distance_between_indices(problem, path[i - 1], path[i])
        #sum+=calculate_time_cost(problem,path)
    return sum


def calculate_solution_distance_dep(problem, solution):
    total_distance = 0.0
    N=len(solution)
    #print(N,'N')
    for i in range(1,N):
        path=solution[i]
        total_distance += calculate_path_distance_dep(problem, path)
        #print(path,'pt')
        #print(total_distance,'tt')
    return total_distance


def calculate_time_series(problem,path):
    n = len(path)
    consumption = [[0 for _ in range(2)]for _ in range(n)]
    VELOCITY = config.Velocity
    for i in range(1, n - 1):
        #print("st")
        #print(consumption[0][0])
        #print(calculate_distance_between_indices(problem,path[i],path[i-1]))
        arrive_time = consumption[i-1][1]+calculate_distance_between_indices(problem,path[i],path[i-1])/VELOCITY
        if arrive_time<problem.get_TW(path[i], 0) :
            if i!=1:
                consumption[i][0] = arrive_time
                consumption[i][1] = problem.get_TW(path[i], 0)
                #consumption[i][1] = arrive_time
            else:
                consumption[i][0] = problem.get_TW(path[i], 0)
                consumption[i][1] = problem.get_TW(path[i], 0)
        else:
            consumption[i][0] = arrive_time
            consumption[i][1] = arrive_time
    #print(path)
    #print(consumption)
    return consumption


def calculate_time_cost(problem, path):
    n = len(path)
    consumption = calculate_time_series(problem, path)
    #print(consumption,'cons')
    TW_cost = 0
    Penalty_wait = 150
    Penalty_late = 150
    for i in range(1, n - 1):
        if consumption[i][0]<problem.get_TW(path[i],0):
            #print("hya")
            TW_cost+=Penalty_wait*(problem.get_TW(path[i],0)-consumption[i][0])
        elif consumption[i][1]>problem.get_TW(path[i],1):
            #print("la")
            #print(consumption[i][1])
            #print(problem.get_TW(path[i],1))
            TW_cost += Penalty_late * (consumption[i][1]-problem.get_TW(path[i], 1))
    #print(TW_cost)
    return TW_cost


def calculate_outside_ratio(problem,solution):
    VELOCITY=config.Velocity
    paras_solution=[]
    for path in solution:
        series=calculate_time_series(problem,path)
        n=len(path)
        if(n==2):
            continue
        paras_path=[series[n-2][1]+calculate_distance_between_indices(problem,path[n-2],path[n-1])/VELOCITY]
        early_time=0.0
        late_time=0.0
        for i in range(n):
            if i==0 or i==n-1:
                continue
            if problem.get_TW(path[i],0)>series[i][0]:
                early_time+= problem.get_TW(path[i],0)-series[i][0]
            if problem.get_TW(path[i],1)<series[i][0]:
                late_time -= problem.get_TW(path[i],1)-series[i][0]
        paras_path.append(early_time)
        paras_path.append(late_time)
        paras_path.append(early_time/paras_path[0])
        paras_path.append(late_time/paras_path[0])
        paras_path.append(paras_path[-1]+paras_path[-2])
        paras_solution.append(paras_path)
    sums=0.0
    print(paras_solution,'ps')
    for pp in paras_solution:
        sums+=pp[-1]
    print(sums/len(paras_solution),'avr')

        





def validate_solution(problem, solution, distance=None):
    if config.problem == 'tsp':
        if len(solution) != 1:
            return False
    visited = [0] * (problem.get_num_customers() + 1)
    for path in solution:
        if path[0] != 0 or path[-1] != 0:
            print(1)
            return False
        consumption = calculate_consumption(problem, path)
        #print(consumption)
        if consumption[-2] > problem.get_capacity(path[0]):
            print(consumption)
            print(problem.get_capacity(path[0]))
            print(2)
            return False
        for customer in path[1:-1]:
            visited[customer] += 1
    for customer in range(1, len(visited)):
        if visited[customer] != 1:
            print(3)
            return False
    if config.problem == 'tsp':
        if visited[0] != 0:
            return False
    if distance is not None and math.fabs(distance - calculate_solution_distance(problem, solution)) > EPSILON:
        print(distance)
        print(calculate_solution_distance(problem,solution))
        return False
    return True


def two_opt(trip, first, second):
    new_trip = copy.deepcopy(trip)
    if first > second:
        first, second = second, first
    first = first + 1
    while first < second:
        temp = copy.copy(new_trip[first])
        new_trip[first] = copy.copy(new_trip[second])
        new_trip[second] = temp
        first = first + 1
        second = second - 1
    return new_trip


def apply_two_opt(trip, distance, top_indices_eval, offset=0):
    #TODO(xingwen): this implementation is very inefficient.
    n = len(trip)
    top_indices = top_indices_eval[0]
    num_indices = len(top_indices)
    min_distance = float('inf')
    min_trip = None
    for i in range(num_indices - 1):
        first = top_indices[i]
        for j in range(i + 1, num_indices):
            second = top_indices[j]
            new_trip = two_opt(trip, (first + offset) % n, (second + offset) % n)
            new_distance = calculate_trip_distance(new_trip)
            if new_distance < min_distance:
                min_distance = new_distance
                min_trip = new_trip
            # print('distance={}, new_distance={}'.format(distance, new_distance))
    if min_distance < distance:
        return min_trip, min_distance
    else:
        return trip, distance


def two_exchange(trip):
    n = len(trip)
    min_delta = -1e-6
    min_first, min_second = None, None
    for first in range(n - 1):
        for second in range(first + 2, min(first + 11, n)):
            if first == 0 and second == n - 1:
                continue
            before = calculate_distance(trip[first - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[first + 1]) \
                    + calculate_distance(trip[second - 1], trip[second]) \
                    + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first - 1], trip[second]) \
                    + calculate_distance(trip[second], trip[first + 1]) \
                    + calculate_distance(trip[second - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[(second + 1) % n])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                min_first = first
                min_second = second
    if min_first is None:
        return trip, calculate_trip_distance(trip)
    else:
        new_trip = copy.deepcopy(trip)
        temp = copy.copy(new_trip[min_first])
        new_trip[min_first] = copy.copy(new_trip[min_second])
        new_trip[min_second] = temp
        return new_trip, calculate_trip_distance(new_trip)


def relocate(trip):
    n = len(trip)
    min_delta = -1e-6
    min_first, min_second = None, None
    for first in range(n):
        for away in range(-10, 10, 1):
            second = (first + away + n) % n
            if second == (first - 1 + n) % n or second == first:
                continue
            before = calculate_distance(trip[first - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[(first + 1) % n]) \
                    + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first - 1], trip[(first + 1) % n]) \
                    + calculate_distance(trip[second], trip[first]) \
                    + calculate_distance(trip[first], trip[(second + 1) % n])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                min_first = first
                min_second = second
    if min_first is None:
        return trip, calculate_trip_distance(trip)
    else:
        new_trip = copy.deepcopy(trip)
        temp = copy.copy(new_trip[min_first])
        to_index = min_first
        while to_index != min_second:
            next_index = (to_index + 1) % n
            new_trip[to_index] = copy.copy(new_trip[next_index])
            to_index = next_index
        new_trip[min_second] = temp
        return new_trip, calculate_trip_distance(new_trip)


def mutate(trip):
    n = len(trip)
    min = -1e-6
    label = None
    for first in range(n - 1):
        for second in range(first + 2, n):
            before = calculate_distance(trip[first], trip[first + 1]) \
                     + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first], trip[second]) \
                    + calculate_distance(trip[first + 1], trip[(second + 1) % n])
            delta = after - before
            if delta < min:
                min = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return two_opt(trip, label[0], label[1]), min, label


def is_illegal_path(problem,path):
    if len(path)>config.max_points_per_trip:
        return True
    capacity_left = problem.get_capacity(0)
    for i in range(0,len(path)):
        if path[i]!=0:
            capacity_left -= problem.get_capacity(path[i])
            if capacity_left < 0:
                return True
    for i in range(0, len(path)):
        if path[i] != 0 and path[i] % 2 == 0 and (path[i]-1) not in path[0:i]:
            return True
    return False


def is_illegal_two_paths(path1, path2):
    path = path1 + path2
    return is_illegal_path(path)


def is_illegal_three_paths(path1, path2, path3):
    path = path1 + path2 + path3
    return is_illegal_path(path)


def is_illegal_solution(problem, solution):
    for path in solution:
        if is_illegal_path(problem, path):
            return True

    return False



def do_two_opt_path(path, first, second):
    improved_path = copy.deepcopy(path)
    first = first + 1
    while first < second:
        improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
        first = first + 1
        second = second - 1
    return improved_path


def two_opt_path(problem, path,last_time_cost):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(n - 1):
        for second in range(first + 2, n):
            before = calculate_distance_between_indices(problem, path[first], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
            after = calculate_distance_between_indices(problem, path[first], path[second]) \
                    + calculate_distance_between_indices(problem, path[first + 1], path[second + 1])
            delta = after - before
            path_tmp = do_two_opt_path(path,first,second)
            delta+=(calculate_time_cost(problem,path_tmp)-last_time_cost)
            if is_illegal_path(problem, path_tmp):
                delta=float('inf')
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_two_opt_path(path, label[0], label[1]), min_delta, label



def do_exchange_path_pair(path, firstp, secondp, firstd, secondd):
    improved_path = copy.deepcopy(path)
    improved_path[firstp], improved_path[firstd],improved_path[secondp], improved_path[secondd] \
            = improved_path[secondp], improved_path[secondd], improved_path[firstp], improved_path[firstd]
    return improved_path



def exchange_path_pair(problem,path,last_time_cost):
    n=len(path) -1
    min_delta = -EPSILON
    label = None
    consumption=calculate_consumption(problem,path)
    if(n<5):
        return None,None,None
    for firstp in range(1,n-1):
        if path[firstp]%2==0:
            continue
        firstd=find_point_index_in_path(path[firstp]+1, path)
        for secondp in range(firstp+1,n-1):
            if path[secondp]%2==0:
                continue
            if problem.get_capacity(path[secondp])+consumption[firstp-1]>problem.get_capacity(0):
                continue
            secondd=find_point_index_in_path(path[secondp]+1,path)
            pathtmp=do_exchange_path_pair(path,firstp,secondp,firstd,secondd)
            if is_illegal_path(problem,pathtmp):
                continue
                delta=float('inf')
            delta=calculate_path_distance(problem,pathtmp)-calculate_path_distance(problem,path)
            if delta < min_delta:
                min_delta = delta
                label = firstp, secondp, firstd, secondd
    if label is None:
        return None, None, None
    else:
        return do_exchange_path_pair(path, label[0], label[1],label[2],label[3]), min_delta, label


def do_exchange_path(path, first, second):
    improved_path = copy.deepcopy(path)
    improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
    return improved_path


def exchange_path(problem, path,last_time_cost):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - 1):
        for second in range(first + 1, n):
            path_tmp=do_exchange_path(path,first,second)
            if is_illegal_path(problem, path_tmp):
                continue
                delta=float('inf')
            delta=calculate_path_distance(problem,path_tmp)-calculate_path_distance(problem,path)
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_exchange_path(path, label[0], label[1]), min_delta, label


def do_relocate_path(path, first, first_tail, second):
    segment = path[first:(first_tail + 1)]
    improved_path = path[:first] + path[(first_tail + 1):]
    if second > first_tail:
        second -= (first_tail - first + 1)
    return improved_path[:(second + 1)] + segment + improved_path[(second + 1):]


def relocate_path(problem, path,last_time_cost, exact_length=1):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - exact_length + 1):
        first_tail = first + exact_length - 1
        for second in range(n):
            if second >= first - 1 and second <= first_tail:
                continue
            path_tmp=do_relocate_path(path,first,first_tail,second)
            if(is_illegal_path(problem,path_tmp)):
                continue
            delta=calculate_path_distance(problem,path_tmp)-calculate_path_distance(problem,path)
            if delta < min_delta:
                min_delta = delta
                label = first, first_tail, second
    if label is None:
        return None, None, None
    else:
        return do_relocate_path(path, label[0], label[1], label[2]), min_delta, label


def calculate_consumption(problem, path):
    n = len(path)
    consumption = [0] * n
    consumption[0] = 0
    for i in range(1, n - 1):
        consumption[i] = consumption[i - 1] + problem.get_capacity(path[i])
    consumption[n - 1] = consumption[n - 2]
    return consumption


def find_path_index_in_solution(path,solution):
    for i in range(0,len(solution)):
        if(solution[i]==path):
            return i
    return -1


def do_cross_two_paths(path_first, path_second, first, second):
    return path_first[:(first + 1)] + path_second[(second + 1):], path_second[:(second + 1)] + path_first[(first + 1):]


def cross_two_paths(problem, path_first, path_second, improved_solution,last_time_cost_first,last_time_cost_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    start_of_second_index = 0
    for first in range(n_first):
        #print(path_first[1:(first+1)])
        if is_illegal_cut(path_first[1:(first+1)]):
            continue
        capacity_from_first_to_second = consumed_capacities_first[n_first - 1] - consumed_capacities_first[first]
        for second in range(start_of_second_index, n_second):
            #print(path_second[1:(second+1)])
            if (first+second)%2 == 1:
                continue
            if is_illegal_cut(path_second[1:(second+1)]):
                continue
            if consumed_capacities_second[second] + capacity_from_first_to_second > problem.get_capacity(path_second[0]):
                continue
            if consumed_capacities_first[first] + (consumed_capacities_second[n_second - 1] - consumed_capacities_second[second]) > problem.get_capacity(path_first[0]):
                #start_of_second_index = second + 1
                continue
            path_first_temp,path_second_temp=do_cross_two_paths(path_first, path_second, first, second)
            if is_illegal_path(problem,path_first_temp) or is_illegal_path(problem, path_second_temp):
                continue
            before = calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_first[first + 1])
            delta = after - before
            delta+=(calculate_time_cost(problem,path_first_temp)-last_time_cost_first+calculate_time_cost(problem,path_second_temp)-last_time_cost_second)
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_cross_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label


def do_relocate_two_paths(path_first, path_second, first, first_tail, second):
    return path_first[:first] + path_first[(first_tail + 1):], \
           path_second[:(second + 1)] + path_first[first:(first_tail + 1)] + path_second[(second + 1):]


def relocate_two_paths(problem, path_first, path_second, improved_solution,last_time_cost_first,last_time_cost_second, exact_length=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    max_length = 1
    min_length = 1
    if exact_length:
        max_length = exact_length
        min_length = exact_length
    for first in range(1, n_first):
        for first_tail in range((first + min_length - 1), min(first + max_length, n_first)):
            capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                break
            for second in range(0, n_second):
                #before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                #     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                #     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
                #after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first_tail + 1])\
                #     + calculate_distance_between_indices(problem, path_second[second], path_first[first])\
                #     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second + 1])
                #delta = after - before
                path_first_temp, path_second_temp = do_relocate_two_paths(path_first, path_second, first, first_tail, second)
                if is_illegal_path(problem, path_first_temp) or is_illegal_path(problem, path_second_temp):
                    continue
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second + 1])
                delta = after - before
                delta+=(calculate_time_cost(problem,path_first_temp)-last_time_cost_first+calculate_time_cost(problem,path_second_temp)-last_time_cost_second)
                if delta < min_delta:
                    min_delta = delta
                    label = first, first_tail, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_relocate_two_paths(path_first, path_second, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, min_delta, label


def is_illegal_cut(cut):
    #print(cut)
    for point in cut:
        if point%2==1 and (point+1) not in cut:
            return True
        elif point%2==0 and (point-1) not in cut:
            return True
    return False


def do_exchange_two_paths(path_first, path_second, first, first_tail, second, second_tail):
    return path_first[:first] + path_second[second:(second_tail + 1)] + path_first[(first_tail + 1):], \
           path_second[:second] + path_first[first:(first_tail + 1)] + path_second[(second_tail + 1):]


def exchange_two_paths(problem, path_first, path_second, improved_solution,last_time_cost_first,last_time_cost_second, exact_lengths=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    if exact_lengths:
        min_length_first, max_length_first = exact_lengths[0], exact_lengths[0]
        min_length_second, max_length_second = exact_lengths[1], exact_lengths[1]
    else:
        min_length_first, max_length_first = 2, 2
        min_length_second, max_length_second = 2, 2

    min_delta = -EPSILON
    label = None
    all_delta = 0.0
    for first in range(1, n_first):
        for first_tail in range((first + min_length_first - 1), min(first + max_length_first, n_first)):
            if first_tail >= n_first:
                break
            if is_illegal_cut(path_first[first:(first_tail+1)]):
                continue
            for second in range(1, n_second):
                if first_tail >= n_first:
                    break
                for second_tail in range((second + min_length_second - 1), min(second + max_length_second, n_second)):
                    if first_tail >= n_first:
                        break
                    if second_tail >= n_second:
                        break
                    if is_illegal_cut(path_second[second:(second_tail+1)]):
                        continue
                    capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1]) - \
                                          (consumed_capacities_second[second_tail] - consumed_capacities_second[second - 1])
                    if consumed_capacities_first[n_first - 1] - capacity_difference <= problem.get_capacity(path_first[0]) and \
                            consumed_capacities_second[n_second - 1] + capacity_difference <= problem.get_capacity(path_second[0]):
                        pass
                    else:
                        continue
                    path_first_temp, path_second_temp = do_exchange_two_paths(path_first, path_second, first, first_tail, second,second_tail)
                    if is_illegal_path(problem,path_first_temp) or is_illegal_path(problem, path_second_temp):
                        continue
                    before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second])\
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_second[second_tail + 1])
                    after = calculate_distance_between_indices(problem, path_first[first - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second_tail + 1])
                    delta = after - before
                    delta+=(calculate_time_cost(problem,path_first_temp)-last_time_cost_first+calculate_time_cost(problem,path_second_temp)-last_time_cost_second)
                    if delta < -EPSILON:
                        all_delta += delta
                        label = first, first_tail, second, second_tail
                        path_first, path_second = do_exchange_two_paths(path_first, path_second, label[0], label[1], label[2], label[3])
                        n_first = len(path_first) - 1
                        n_second = len(path_second) - 1
                        consumed_capacities_first = calculate_consumption(problem, path_first)
                        consumed_capacities_second = calculate_consumption(problem, path_second)
                        last_time_cost_first=calculate_time_cost(problem,path_first)
                        last_time_cost_second=calculate_time_cost(problem,path_second)
    if label is None:
        return None, None, None, None
    else:
        return path_first, path_second, all_delta, label


def find_point_index_in_path(point,path):
    for i in range(0,len(path)):
        if(path[i]==point):
            return i
    return -1


def find_path_with_point(point,solution):
    for i in range(0,len(solution)):
        if(point in solution[i]):
            return i
    return -1


def detect_best_point(problem, path, path_point_index):
    min_delta = float("inf")
    label = None
    consumed_capacities = calculate_consumption(problem, path)
    num_sample_points = get_num_points(config)
    for point in range(0,num_sample_points):
        if point %2 ==1 and point not in path:
            delta = calculate_distance_between_indices(problem, path[path_point_index], point)
            path_tmp=path[:path_point_index+1]+[point]+path[path_point_index+1:]
            delta+=(calculate_time_cost(problem,path_tmp)-calculate_time_cost(problem,path))
            if delta < min_delta and consumed_capacities[path_point_index]+problem.get_capacity(point)<problem.get_capacity(path[0]):
                min_delta=delta
                label=point
    if label is None:
        return None
    return label


def detect_best_insert_loc(problem, best_point_index, delivery_index, start_index, path,path_to_exchange):
    best_distance = calculate_distance_between_indices(problem, path_to_exchange[best_point_index],path_to_exchange[delivery_index])+calculate_distance_between_indices(problem, path[start_index],path_to_exchange[delivery_index])
    #print(path)
    path_tmp=path[:start_index]+[path_to_exchange[best_point_index]]+[path_to_exchange[delivery_index]]+path[start_index:]
    #print(path_tmp,'btmp')
    best_distance +=(calculate_time_cost(problem,path_tmp)-calculate_time_cost(problem,path))
    insert_index = start_index
    for i in range(start_index,len(path)-1):
        new_distance = calculate_distance_between_indices(problem,path[i],path_to_exchange[delivery_index])+calculate_distance_between_indices(problem, path[i+1],path_to_exchange[delivery_index])
        #print("new"+str(new_distance))
        path_tmp=path[:start_index]+[path_to_exchange[best_point_index]]+path[start_index:i+1]+[path_to_exchange[delivery_index]]+path[i+1:]
        new_distance+=(calculate_time_cost(problem,path_tmp)-calculate_time_cost(problem,path))
        if new_distance<best_distance:
            best_distance=new_distance
            insert_index = i+1
    #print(insert_index,'idx')
    return insert_index,best_distance
    
    
def do_best_point_exchange(path_first, path_to_exchange, best_point_index, delivery_index, first, insert_index):
    ret1=path_first[:(first+1)]+[path_to_exchange[best_point_index]]+path_first[(first+1):insert_index]+ \
                 [path_to_exchange[delivery_index]]+path_first[insert_index:]
    ret2=path_to_exchange[:best_point_index]+path_to_exchange[(best_point_index+1):delivery_index]+path_to_exchange[(delivery_index+1):]
    return path_first[:(first+1)]+[path_to_exchange[best_point_index]]+path_first[(first+1):insert_index]+ \
                 [path_to_exchange[delivery_index]]+path_first[insert_index:],\
           path_to_exchange[:best_point_index]+path_to_exchange[(best_point_index+1):delivery_index]+path_to_exchange[(delivery_index+1):]

    
    
def best_point_exchange(problem, path_first,solution,last_time_cost):
    n_first = len(path_first) - 1
    label = None
    min_delta = -EPSILON
    for first in range(0, n_first):
        best_point = detect_best_point(problem,path_first,first)
        #print(best_point)
        if best_point is None:
            continue
        #print(best_point)
        path_to_exchange_index = find_path_with_point(best_point,solution)
        path_to_exchange=solution[path_to_exchange_index]
        #print(path_to_exchange)
        #path_to_exchange_index = find_path_index_in_solution(path_to_exchange,solution)
        best_point_index = find_point_index_in_path(best_point, path_to_exchange)
        delivery_index=find_point_index_in_path(best_point+1, path_to_exchange)
        insert_index,best_distance=detect_best_insert_loc(problem, best_point_index, delivery_index, first+1, path_first,path_to_exchange)
        #next1=0
        #next2=0
        #if delivery_index==best_point_index+1:
        #    next1=1
        #if insert_index==first+1:
        #    next2=1
        improved_path_first_tmp, improved_path_second_tmp = do_best_point_exchange(path_first,path_to_exchange,best_point_index,delivery_index,first,insert_index)
        if is_illegal_path(problem, improved_path_first_tmp):
            continue
        if delivery_index==best_point_index+1:
            before1=calculate_distance_between_indices(problem, path_to_exchange[best_point_index-1], path_to_exchange[best_point_index])\
                    + calculate_distance_between_indices(problem, path_to_exchange[best_point_index], path_to_exchange[delivery_index])\
                    + calculate_distance_between_indices(problem, path_to_exchange[delivery_index], path_to_exchange[delivery_index+1])
            after1=calculate_distance_between_indices(problem, path_to_exchange[best_point_index-1], path_to_exchange[delivery_index+1])
        else:
            before1 = calculate_distance_between_indices(problem, path_to_exchange[best_point_index - 1],path_to_exchange[best_point_index]) \
                      + calculate_distance_between_indices(problem, path_to_exchange[best_point_index],path_to_exchange[best_point_index+1]) \
                      + calculate_distance_between_indices(problem, path_to_exchange[delivery_index],path_to_exchange[delivery_index + 1]) \
                      + calculate_distance_between_indices(problem, path_to_exchange[delivery_index],path_to_exchange[delivery_index - 1])
            after1=calculate_distance_between_indices(problem, path_to_exchange[best_point_index-1], path_to_exchange[best_point_index+1])+ \
                   calculate_distance_between_indices(problem, path_to_exchange[delivery_index - 1],path_to_exchange[delivery_index + 1])
        if insert_index==first+1:
            before2=calculate_distance_between_indices(problem, path_first[first], path_first[insert_index])
            after2=calculate_distance_between_indices(problem, path_first[first], path_to_exchange[best_point_index])\
                   +calculate_distance_between_indices(problem, path_to_exchange[best_point_index], path_to_exchange[delivery_index])\
                   +calculate_distance_between_indices(problem, path_to_exchange[delivery_index], path_first[insert_index])
        else:
            before2 = calculate_distance_between_indices(problem, path_first[first], path_first[first+1])\
                      +calculate_distance_between_indices(problem, path_first[insert_index-1], path_first[insert_index])
            after2 = calculate_distance_between_indices(problem, path_first[first], path_to_exchange[best_point_index])\
                     +calculate_distance_between_indices(problem, path_to_exchange[best_point_index], path_first[first+1])\
                     +calculate_distance_between_indices(problem, path_first[insert_index-1], path_to_exchange[delivery_index])\
                     +calculate_distance_between_indices(problem, path_to_exchange[delivery_index], path_first[insert_index])
        delta = after1+after2-before1-before2
        delta += (calculate_time_cost(problem,improved_path_first_tmp)-last_time_cost+calculate_time_cost(problem,improved_path_second_tmp)-calculate_time_cost(problem,path_to_exchange))
        #if is_illegal_path(problem, improved_path_first_tmp):
            #print("salaibe")
        #    delta = float('inf')
        if delta < min_delta:
            #print("nice"+str(first)+" "+str(insert_index))
            min_delta = delta
            label = path_to_exchange,best_point_index,delivery_index,first,insert_index,path_to_exchange_index
        #else:
            #print("sorry"+str(delta))
    if label is None:
            return None, None, None, None,None
    else:
        improved_path_first, improved_path_to_exchange= do_best_point_exchange(path_first,label[0],label[1],label[2],label[3],label[4])
        #print("min"+str(min_delta))
        return improved_path_first, improved_path_to_exchange, label[5], min_delta, label


def do_eject_two_paths(path_first, path_second, first, second):
    return path_first[:first] + path_first[(first + 1):], \
           path_second[:second] + path_first[first:(first + 1)] + path_second[(second + 1):]


def eject_two_paths(problem, path_first, path_second, improved_solution):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities_second = calculate_consumption(problem, path_second)

    for first in range(1, n_first):
        for second in range(1, n_second):
            capacity_difference = problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                continue
            before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1])
            delta = after - before
            path_first_temp,path_second_temp=do_eject_two_paths(path_first, path_second, first, second)
            improved_solution_tmp = copy.deepcopy(improved_solution)
            path_index_first = find_path_index_in_solution(path_first, improved_solution)
            path_index_second = find_path_index_in_solution(path_second, improved_solution)
            improved_solution_tmp[path_index_first] = path_first_temp
            improved_solution_tmp[path_index_second] = path_second_temp
            #if is_illegal_solution(improved_solution_tmp):
            #    delta=float('inf')
            #if is_illegal_two_paths(path_first_temp, path_second_temp):
            #    delta=float('inf')
            if is_illegal_path(problem, path_first_temp) or is_illegal_path(problem, path_second_temp):
                delta=float('inf')
            if delta < min_delta:
                min_delta = delta
                label = first, second, path_second[second]
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_eject_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label[2]


def insert_into_path(path, first):
    n = len(path) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities = calculate_consumption(problem, path)

    if consumed_capacities[n - 1] + problem.get_capacity(first) > problem.get_capacity(path[0]):
        return None, None, None
    for second in range(0, n):
        before = calculate_distance_between_indices(problem, path[second], path[second + 1])
        after = calculate_distance_between_indices(problem, path[second], first) \
                + calculate_distance_between_indices(problem, first, path[second + 1])
        delta = after - before
        if delta < min_delta:
            min_delta = delta
            label = second

    improved_path_third = path[:(label + 1)] + [first] + path[(label + 1):]
    return improved_path_third, min_delta, label


def do_eject_three_paths(path_first, path_second, path_third, first, second, third):
    return path_first[:first] + [path_third[third]] + path_first[(first + 1):], \
           path_second[:second] + [path_first[first]] + path_second[(second + 1):], \
           path_third[:third] + [path_second[second]] + path_third[(third + 1):]


def eject_three_paths(problem, path_first, path_second, path_third, improved_solution):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    n_third = len(path_third) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    consumed_capacities_third = calculate_consumption(problem, path_third)

    for first in range(1, n_first):
        for second in range(1, n_second):
            if consumed_capacities_second[n_second - 1] + problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second]) > problem.get_capacity(path_second[0]):
                continue
            for third in range(1, n_third):
                if consumed_capacities_third[n_third - 1] + problem.get_capacity(path_second[second]) - problem.get_capacity(path_third[third]) > problem.get_capacity(path_third[0]):
                    continue
                if consumed_capacities_first[n_first - 1] + problem.get_capacity(path_third[third]) - problem.get_capacity(path_first[first]) > problem.get_capacity(path_first[0]):
                    continue
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_third[third + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_third[third + 1])
                delta = after - before
                path_first_temp, path_second_temp, path_third_temp = do_eject_three_paths(path_first, path_second, path_third, first, second, third)
                improved_solution_tmp = copy.deepcopy(improved_solution)
                path_index_first = find_path_index_in_solution(path_first, improved_solution)
                path_index_second = find_path_index_in_solution(path_second, improved_solution)
                path_index_third = find_path_index_in_solution(path_third, improved_solution)
                improved_solution_tmp[path_index_first] = path_first_temp
                improved_solution_tmp[path_index_second] = path_second_temp
                improved_solution_tmp[path_index_third] = path_third_tmp
                #if is_illegal_solution(improved_solution_tmp):
                #    delta=float('inf')
                #if is_illegal_three_paths(path_first_temp, path_second_temp, path_third_temp):
                #    delta = float('inf')
                if is_illegal_path(problem, path_first_temp) or is_illegal_path(problem, path_second_temp) or is_illegal_path(problem, path_third_temp):
                    delta = float('inf')
                if delta < min_delta:
                    min_delta = delta
                    label = first, second, third
                    improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
                        path_first, path_second, path_third, label[0], label[1], label[2])
                    return improved_path_first, improved_path_second, improved_path_third, min_delta, label
    if label is None:
        return None, None, None, None, None
    else:
        improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
            path_first, path_second, path_third, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, improved_path_third, min_delta, label


def improve_solution(problem, solution):
    improved_solution = copy.deepcopy(solution)
    all_delta = 0.0
    num_paths = len(improved_solution)

    for path_index in range(num_paths):
        improved_path, delta, label = two_opt_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

        improved_path, delta, label = exchange_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

        improved_path, delta, label = relocate_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

    for path_index_first in range(num_paths - 1):
        for path_index_second in range(path_index_first + 1, num_paths):
            improved_path_first, improved_path_second, delta, label = cross_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution)
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta

            improved_path_first, improved_path_second, delta, label = exchange_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution)
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta

            improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution)
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta
            improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                problem, improved_solution[path_index_second], improved_solution[path_index_first], improved_solution)
            if label:
                improved_solution[path_index_first] = improved_path_second
                improved_solution[path_index_second] = improved_path_first
                all_delta += delta
    return improved_solution, all_delta


def  get_exact_lengths_for_exchange_two_paths(action):
    if action in range(12, 16):
        exact_lengths = [
            [2, 2],
            [2, 4],
            [4, 2],
            [4, 4],
        ]
        return exact_lengths[action - 12]
    else:
        return None


def improve_solution_by_action(step, problem, solution, action,path_aim_1,path_aim_2,just_test=False):
    improved_solution = copy.deepcopy(solution)
    all_delta = 0.0
    num_paths = len(improved_solution)


    if action in [1, 2, 3, 4,5,6,11, 16] or config.problem == 'tsp':
        path_index=path_aim_1
        
        if not just_test:
            modified = problem.should_try(action, path_index)
        else:
            modified = True
        if modified:
            last_time_cost = calculate_time_cost(problem,improved_solution[path_index])
        while modified:
            if action == 1:
                improved_path, delta, label = two_opt_path(problem, improved_solution[path_index],last_time_cost)
            elif action == 2 :
                improved_path, delta, label = exchange_path(problem, improved_solution[path_index],last_time_cost)
            elif action==16:
                improved_path,delta,label = exchange_path_pair(problem,improved_solution[path_index],last_time_cost)
            elif action==11:
                improved_path,improved_path_to_exchange,exchange_index,delta,label=best_point_exchange(problem,improved_solution[path_index],improved_solution,last_time_cost)
                    
            else:
                exact_lengths = {
                    3: 1,
                    4: 2,
                    5: 3,
                    6: 4,
                }
                improved_path, delta, label = relocate_path(problem, improved_solution[path_index],last_time_cost, exact_length=exact_lengths[action])
            if label and action!=11:
                modified = True
                if not just_test:
                    problem.mark_change_at(step, [path_index])
                improved_solution[path_index] = improved_path
                last_time_cost = calculate_time_cost(problem,improved_path)
                all_delta += delta
            elif label :
                modefied=True
                if not just_test:
                    problem.mark_change_at(step,[path_index,exchange_index])
                improved_solution[path_index] = improved_path
                improved_solution[exchange_index] = improved_path_to_exchange
                last_time_cost = calculate_time_cost(problem,improved_path)
                all_delta+=delta
            else:
                modified = False
                if not just_test:
                    problem.mark_no_improvement(step, action, path_index)
        return improved_solution, all_delta

    path_index_first=path_aim_1
    path_index_second=path_aim_2
    if not just_test:
        modified = problem.should_try(action, path_index_first, path_index_second)
    else:
        modified = True
    if modified:
        last_time_cost_first=calculate_time_cost(problem,improved_solution[path_index_first])
        last_time_cost_second=calculate_time_cost(problem,improved_solution[path_index_second])
    if action in ([7] + list(range(12, 16))):
        while modified:
            if action == 7:
                improved_path_first, improved_path_second, delta, label = cross_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution,last_time_cost_first,last_time_cost_second)
            else:
                improved_path_first, improved_path_second, delta, label = exchange_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution,last_time_cost_first,last_time_cost_second, get_exact_lengths_for_exchange_two_paths(action))
            if label :
                modified = True
                if not just_test:
                    problem.mark_change_at(step, [path_index_first, path_index_second])
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                last_time_cost_first=calculate_time_cost(problem,improved_path_first)
                last_time_cost_second=calculate_time_cost(problem,improved_path_second)
                #print(last_time_cost_first,'ltcf')
                #print(last_time_cost_second,'ltcs')
                #print(delta,'dt')
                all_delta += delta
                #print(all_delta,'adt')
            else:
                modified = False
                if not just_test:
                    problem.mark_no_improvement(step, action, path_index_first, path_index_second)
    times=0
    while action in [8, 9, 10] and modified:
        modified = False
        times+=1
        if times>10:
            scio.savemat('relocate_loop.mat',{'step':step})
            print(delta,'d')
            print(all_delta,'ad')
            print(improved_solution,'is')
            return solution,0.0
        improved_path_first, improved_path_second, delta, label = relocate_two_paths(
            problem, improved_solution[path_index_first], improved_solution[path_index_second], improved_solution, last_time_cost_first,last_time_cost_second, action - 7)
        if label :
            modified = True
            if not just_test:
                problem.mark_change_at(step, [path_index_first, path_index_second])
            improved_solution[path_index_first] = improved_path_first
            improved_solution[path_index_second] = improved_path_second
            last_time_cost_first=calculate_time_cost(problem,improved_path_first)
            last_time_cost_second=calculate_time_cost(problem,improved_path_second)

            all_delta += delta
            improved_path_first, improved_path_second, delta, label = relocate_two_paths(problem, improved_solution[path_index_second], improved_solution[path_index_first], improved_solution,last_time_cost_second,last_time_cost_first, action - 7)
        if label :
            modified = True
            if not just_test:
                problem.mark_change_at(step, [path_index_first, path_index_second])
            improved_solution[path_index_first] = improved_path_second
            improved_solution[path_index_second] = improved_path_first
            last_time_cost_first=calculate_time_cost(problem,improved_path_second)
            last_time_cost_second=calculate_time_cost(problem,improved_path_first)
            all_delta += delta
        if not modified:
            if not just_test:
                problem.mark_no_improvement(step, action, path_index_first, path_index_second)

    return improved_solution, all_delta


def dense_to_one_hot(labels_dense, num_training_points):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * config.num_training_points
  labels_one_hot = np.zeros((num_labels, config.num_training_points))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def reshape_input(input, x, y, z):
    return np.reshape(input, (x, y, z))


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def build_multi_operator_model(raw_input):
    input_sequence = tf.unstack(raw_input, config.num_training_points, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(config.num_lstm_units, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(config.num_lstm_units, forget_bias=1.0)
    lstm_output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_sequence, dtype=tf.float32)
    layer_1 = lstm_output[-1]

    layer_2 = tf.contrib.layers.fully_connected(layer_1, config.num_feedforward_units, activation_fn=tf.nn.relu)
    layer_2 = tf.contrib.layers.fully_connected(layer_2, config.num_lstm_units, activation_fn=None)
    layer_2 += layer_1
    layer_2 = tf.contrib.layers.batch_norm(layer_2, is_training=is_training)
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, keep_prob)

    output_layer = tf.contrib.layers.fully_connected(layer_2, config.num_training_points, activation_fn=None)
    return output_layer


def get_random_capacities(n):
    capacities = [0] * n
    if config.problem == 'vrp':
        depot_capacity_map = {
            10: 20,
            50: 40,
            100: 50
        }
        capacities[0] = depot_capacity_map.get(n - 1, 50)
        for i in range(1, n):
            if i%2==1:
                capacities[i] = np.random.randint(9) + 1
            else:
                capacities[i]=-capacities[i-1]
    if config.baseline_exp:
        #capacities= [10, 3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3]
        '''
        capacities = [10, 3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3,
              3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3,
              3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3,
              3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3,
              3, -3, 2, -2, 4, -4, 8, -8, 6, -6, 7, -7, 4, -4, 3, -3]
        '''
        #capacities= [10,2,-2,8,-8,5,-5,1,-1,10,-10,5,-5,9,-9,9,-9,3,-3,5,-5,6,-6,6,-6,2,-2,8,-8,2,-2,2,-2,6,-6,3,-3,8,-8,7,-7,2,-2,5,-5,3,-3,4,-4,3,-3]
        
    capacities=[0 for i in range(n)]
    capacities[0]=10
    for i in range(1,n):
        if(i%2):
            capacities[i]=np.random.randint(1,10)
        else:
            capacities[i]=-capacities[i-1]
              
    return capacities


def get_time_windows(n):
    #TWs=[(0,10000), (9, 13), (14, 15), (1, 2), (6, 9), (2, 3), (9, 11), (1, 5), (9, 13), (3, 6), (11, 12), (7, 8), (14, 16), (4, 7), (13, 14), (6, 10), (7, 10), (3, 6), (13, 14), (3, 7), (11, 14), (15, 19), (10, 12), (9, 10), (5, 9), (13, 14), (11, 12), (6, 8), (5, 9), (3, 5), (9, 12), (12, 16), (14, 17), (0, 3), (0, 3), (0, 1), (6, 8), (2, 4), (2, 5), (14, 16), (9, 10), (13, 17), (4, 7), (4, 5), (6, 7), (11, 12), (0, 4), (2, 6), (1, 4), (6, 9), (1, 3)]

    #TWs=[(0,10000), (9, 13), (15, 16), (1, 2), (3, 6), (2, 3), (6, 8), (1, 5), (7, 11), (3, 6), (7, 8), (7, 8), (11, 13), (4, 7), (8, 9), (6, 10), (13, 16), (3, 6), (7, 8), (3, 7), (9, 12), (15, 19), (22, 24), (9, 10), (11, 15), (13, 14), (17, 18), (6, 8), (9, 13), (3, 5), (7, 10), (12, 16), (19, 22), (0, 3), (5, 8), (0, 1), (4, 6), (2, 4), (6, 9), (14, 16), (17, 18), (13, 17), (18, 21), (4, 5), (7, 8), (11, 12), (13, 17), (2, 6), (9, 12), (6, 9), (10, 12)]
     
    TWs=[[0 for i in range(2)]for j in range(n)]
    TWs[0][0]=0
    TWs[0][1]=23
    for i in range(1,n):
        #print(i)
        if i%2:
            TWs[i][0]=np.random.randint(1,15)
            TWs[i][1]=np.random.randint(1,3)+TWs[i][0]
        else:
            TWs[i][0]=TWs[i-1][1]+np.random.randint(1,3)
            TWs[i][1] = np.random.randint(1, 3) + TWs[i][0]
            if(TWs[i][1]>=23):
                TWs[i][1]=22
    
    return TWs


def sample_next_index(to_indices, adjusted_distances):
    if len(to_indices) == 0:
        return 0
    adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
    adjusted_probabilities /= np.sum(adjusted_probabilities)
    return np.random.choice(to_indices, p=adjusted_probabilities)
    # return to_indices[np.argmax(adjusted_probabilities)]


def calculate_replacement_cost(problem, from_index, to_indices):
    return problem.get_distance(from_index, to_indices[0]) + problem.get_distance(from_index, to_indices[2]) \
        - problem.get_distance(to_indices[1], to_indices[0]) - problem.get_distance(to_indices[1], to_indices[2])


class Graph:
    def __init__(self, problem, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for from_index in range(self.num_nodes):
            for to_index in range(from_index + 1, self.num_nodes):
                self.distance_matrix[from_index][to_index] = calculate_replacement_cost(problem, nodes[from_index][1], nodes[to_index])
                self.distance_matrix[to_index][from_index] = calculate_replacement_cost(problem, nodes[to_index][1], nodes[from_index])

    def find_negative_cycle(self):
        distance = [float('inf')] * self.num_nodes
        predecessor = [None] * self.num_nodes
        source = 0
        distance[source] = 0.0

        for i in range(1, self.num_nodes):
            improved = False
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    w = self.distance_matrix[u][v]
                    if distance[u] + w < distance[v]:
                        distance[v] = distance[u] + w
                        predecessor[v] = u
                        improved = True
            if not improved:
                break

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                w = self.distance_matrix[u][v]
                if distance[u] + w + EPSILON < distance[v]:
                    visited = [0] * self.num_nodes
                    negative_cycle = []
                    negative_cycle.append(self.nodes[v][-2:])
                    count = 1
                    while (u != v) and (not visited[u]):
                        negative_cycle.append(self.nodes[u][-2:])
                        visited[u] = count
                        count += 1
                        u = predecessor[u]
                    if u != v:
                        negative_cycle = negative_cycle[visited[u]:]
                    return negative_cycle[::-1], -1.0

        num_cyclic_perturb = 4
        cutoff = 0.3
        if self.num_nodes >= num_cyclic_perturb:
            candidate_cycles = []
            for index in range(self.num_nodes):
                candidate_cycles.append(([index], 0.0))
            for index_to_choose in range(1, num_cyclic_perturb):
                next_candidate_cycles = []
                for cycle in candidate_cycles:
                    nodes = cycle[0]
                    total_distance = cycle[1]
                    for index in range(self.num_nodes):
                        if index not in nodes:
                            if index_to_choose == num_cyclic_perturb - 1:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index] + self.distance_matrix[index][nodes[0]]
                            else:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index]
                            if new_total_distance < cutoff:
                                next_candidate_cycles.append((nodes + [index], new_total_distance))
                candidate_cycles = next_candidate_cycles
            # if len(candidate_cycles):
            #     print('count={}'.format(candidate_cycles))
            if len(candidate_cycles) > 0:
                random_indices = np.random.choice(range(len(candidate_cycles)), 1)[0]
                random_indices = candidate_cycles[random_indices][0]
                negative_cycle = []
                for u in random_indices:
                    negative_cycle.append(self.nodes[u][-2:])
                return negative_cycle, 1.0
        return None, None


def construct_graph(problem, solution, capacity):
    nodes = []
    for path_index in range(len(solution)):
        path = solution[path_index]
        if len(path) > 2:
            node_index = 1
            while node_index < len(path) - 1:
                node_index_end = node_index + 1
                if problem.get_capacity(path[node_index]) == capacity:
                    while problem.get_capacity(path[node_index_end]) == capacity:
                        node_index_end += 1
                    sampled_node_index = np.random.choice(range(node_index, node_index_end))
                    nodes.append([path[sampled_node_index - 1], path[sampled_node_index], path[sampled_node_index + 1],
                                  path_index, sampled_node_index])
                node_index = node_index_end
    graph = Graph(problem, nodes)
    return graph


def get_path_from_cycle(cycle):
    index_of_0 = 0
    while True:
        if cycle[index_of_0] == 0:
            break
        else:
            index_of_0 += 1
    path = cycle[index_of_0:] + cycle[:(index_of_0 + 1)]
    return path


def get_cycle_from_path(path):
    return path[1:]


def construct_solution(problem, existing_solution=None, step=0):
    solution = []
    n = problem.get_num_customers()
    customer_indices = list(range(n + 1))
    if config.problem == 'tsp':
        num_customers = n + 1
        if existing_solution is None:
            cycle = np.random.permutation(num_customers).tolist()
            path = get_path_from_cycle(cycle)
        else:
            num_customers_to_shuffle = min(config.max_num_customers_to_shuffle, num_customers)
            start_index = np.random.randint(num_customers)
            indices_permuted = np.random.permutation(num_customers_to_shuffle)
            cycle = get_cycle_from_path(existing_solution[0])
            cycle_perturbed = copy.copy(cycle)
            for index in range(num_customers_to_shuffle):
                to_index = start_index + index
                if to_index >= num_customers:
                    to_index -= num_customers
                from_index = start_index + indices_permuted[index]
                if from_index >= num_customers:
                    from_index -= num_customers
                cycle_perturbed[to_index] = cycle[from_index]
            path = get_path_from_cycle(cycle_perturbed)
        solution.append(path)
        problem.reset_change_at_and_no_improvement_at()
        return solution

    if (existing_solution is not None) and (config.num_paths_to_ruin != -1):
        distance = calculate_solution_distance(problem, existing_solution)
        min_reconstructed_distance = float('inf')
        solution_to_return = None
        # for _ in range(10):
        for _ in range(1):
            randtmp=np.random.uniform()
            if randtmp<0.05:
                reconstructed_solution = reconstruct_solution(problem, existing_solution, step)
            elif randtmp<0.50:
                #reconstructed_solution = reconstruct_solution_by_pair_exchange(problem,existing_solution,step)
                reconstructed_solution = three_route_exchange(problem,existing_solution,step)
            else:
                for _ in range(2):
                    reconstructed_solution = one_route_exchange(problem,existing_solution,step)
            reconstructed_distance = calculate_solution_distance(problem, reconstructed_solution)
            if reconstructed_distance / distance <= 1.05:
                solution_to_return = reconstructed_solution
                break
            else:
                if reconstructed_distance < min_reconstructed_distance:
                    min_reconstructed_distance = reconstructed_distance
                    solution_to_return = reconstructed_solution
        return solution_to_return
    else:
        start_customer_index = 1

    #trip = [0]
    #capacity_left = problem.get_capacity(0)
    #i = start_customer_index
    #while i <= n:
    #    random_index = np.random.randint(low=i, high=n+1)

        # if len(trip) > 1:
        #     min_index, min_distance = random_index, float('inf')
        #     for j in range(i, n + 1):
        #         if problem.get_capacity(customer_indices[j]) > capacity_left:
        #             continue
        #         distance = calculate_distance_between_indices(problem, trip[-1], customer_indices[j])
        #         if distance < min_distance:
        #             min_index, min_distance = j, distance
        #     random_index = min_index

        # if len(trip) > 1:
        #     min_index, min_distance = 0, calculate_adjusted_distance_between_indices(problem, trip[-1], 0)
        # else:
        #     min_index, min_distance = random_index, float('inf')
        # for j in range(i, n + 1):
        #     if problem.get_capacity(customer_indices[j]) > capacity_left:
        #         continue
        #     distance = calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j])
        #     if distance < min_distance:
        #         min_index, min_distance = j, distance
        # random_index = min_index

     #   to_indices = []
     #   adjusted_distances = []
        # if len(trip) > 1:
        #     to_indices.append(0)
        #     adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], 0))
     #   for j in range(i, n + 1):
     #       if problem.get_capacity(customer_indices[j]) > capacity_left:
     #           continue
     #       if customer_indices[j] > 0 and customer_indices[j]%2 == 0 and ((customer_indices[j]-1) not in customer_indices[0:i]):
     #           continue
     #       to_indices.append(j)
     #       adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
    #    random_index = sample_next_index(to_indices, adjusted_distances)

    #    if random_index == 0 or capacity_left < problem.get_capacity(customer_indices[random_index]):
    #        trip.append(0)
    #        solution.append(trip)
    #        trip = [0]
    #        capacity_left = problem.get_capacity(0)
    #        continue
    #    customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
    #    trip.append(customer_indices[i])
    #    capacity_left -= problem.get_capacity(customer_indices[i])
    #    i += 1
    #if len(trip) > 1:
    #    trip.append(0)
    #    solution.append(trip)
    #solution.append([0, 0])
    #print(solution)
    path_num=config.num_paths
    path=[]
    for i in range(0,path_num):
        solution.append(copy.deepcopy(path))
        solution[i].append(0)
    for j in range(1,n):
        if j%2==1:
            aim_path=(int)((j%(2*path_num)-1)/2)
            solution[aim_path].append(j)
            solution[aim_path].append(j+1)
    for i in range(0,path_num):
        solution[i].append(0)
    solution_to_return = reconstruct_solution(problem,solution,0,path_num,True)

    problem.reset_change_at_and_no_improvement_at()
    #print(solution)
    return solution_to_return


def reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined):
    path0 = copy.deepcopy(existing_solution[paths_ruined[0]])
    path1 = copy.deepcopy(existing_solution[paths_ruined[1]])
    num_exchanged = 0
    for i in range(1, len(path0) - 1):
        for j in range(1, len(path1) - 1):
            if problem.get_capacity(path0[i]) == problem.get_capacity(path1[j]):
                #TODO
                if problem.get_distance(path0[i], path1[j]) < 0.2:
                    path0[i], path1[j] = path1[j], path0[i]
                    num_exchanged += 1
                    break
    if num_exchanged >= 0:
        return [path0, path1]
    else:
        return []



def three_route_exchange(problem,existing_solution,step):
    #print(existing_solution,'ex')
    improved_solution=copy.deepcopy(existing_solution)
    paths=np.random.choice(existing_solution,3,replace=False)
    all_zeros=[1,1,1]
    for i in range(len(paths)):
        if len(paths[i])==2:
            all_zeros[i] = 0
    if sum(all_zeros)==0:
        return existing_solution
    path_first=paths[0]
    path_second=paths[1]
    path_third=paths[2]
    #print(path_first,'f')
    #print(path_second,'s')
    #print(path_third,'t')
    firstp=0
    secondp=0
    thirdp=0
    while(firstp%2==0 and all_zeros[0]!=0):
        firstp=np.random.choice(path_first)
    while(secondp%2==0and all_zeros[1]!=0):
        secondp=np.random.choice(path_second)
    while(thirdp%2==0and all_zeros[2]!=0):
        thirdp=np.random.choice(path_third)
    firstd=find_point_index_in_path(firstp+1,path_first)
    firstp=find_point_index_in_path(firstp,path_first)
    secondd=find_point_index_in_path(secondp+1,path_second)
    secondp=find_point_index_in_path(secondp,path_second)
    thirdd=find_point_index_in_path(thirdp+1,path_third)
    thirdp=find_point_index_in_path(thirdp,path_third)
    path_second_new,isuccess = eject_insert(problem,path_first,path_second,all_zeros[0],all_zeros[1],firstp,secondp,firstd,secondd)
    if isuccess==0:
        return existing_solution
    path_third_new,isuccess = eject_insert(problem,path_second,path_third,all_zeros[1],all_zeros[2],secondp,thirdp,secondd,thirdd)
    if isuccess==0:
        return existing_solution
    path_first_new,isuccess = eject_insert(problem,path_third,path_first,all_zeros[2],all_zeros[0],thirdp,firstp,thirdd,firstd)
    if isuccess==0:
        return existing_solution
    #if(is_illegal_path(problem,path_first_new)or is_illegal_path(problem,path_second_new)or is_illegal_path(problem,path_third_new)):
    #    print(path_first_new,'npfn')
    #    print(path_second_new,'npsn')
    #    print(path_third_new,'nptn')
    #    print("nop")
    #    return existing_solution
    #print(path_first_new,'nf')
    #print(path_second_new,'ns')
    #print(path_third_new,'nt')
    if(path_first!=path_first_new):
        index_first=find_path_index_in_solution(path_first,existing_solution)
        improved_solution[index_first]=copy.deepcopy(path_first_new)
        problem.mark_change_at(step,[index_first])
    if(path_second!=path_second_new):
        index_second=find_path_index_in_solution(path_second,existing_solution)
        improved_solution[index_second]=copy.deepcopy(path_second_new)
        problem.mark_change_at(step,[index_second])
    if(path_third!=path_third_new):
        index_third=find_path_index_in_solution(path_third,existing_solution)
        improved_solution[index_third]=copy.deepcopy(path_third_new)
        problem.mark_change_at(step,[index_third])
    #problem.mark_change_at(step,[index_second,index_first,index_third])
    #print("threeipv")
    return improved_solution


def eject_insert(problem,path_first,path_second,all_zeros_first,all_zeros_second,firstp,secondp,firstd,secondd):
    if(all_zeros_first==0 and all_zeros_second!=0):
        #print("n0w1hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        path_second_new = path_second[:secondp]+path_second[(secondp+1):secondd]+path_second[(secondd+1):]
    elif (all_zeros_second == 0 and all_zeros_first!=0):
        #print("w0n1hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        path_second_new = [0, path_first[firstp], path_first[firstd], 0]
    elif (all_zeros_first==0 and all_zeros_second==0):
        #print("q0q1hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        path_second_new=path_second
    else:
        diff= problem.get_capacity(path_first[firstp]) - problem.get_capacity(path_second[secondp])
        if(diff<=0):
            path_second_new = path_second[:secondp]+[path_first[firstp]]+path_second[(secondp+1):secondd]+[path_first[firstd]]+path_second[(secondd+1):]
        else:
            consumption = calculate_consumption(problem, path_second)
            if (consumption[secondp] + diff > problem.get_capacity(0)):
                return path_second,0
            for index in range(secondp + 1, secondd + 1):
                if (consumption[index] + diff > problem.get_capacity(0)):
                    break
            #if(index!=secondp+1 and index!=secondd):
            #    print("now")
            path_second_new = path_second[:secondp] + [path_first[firstp]] + path_second[(secondp + 1): index] + [
                path_first[firstd]] + path_second[index:secondd] + path_second[(secondd + 1):]
    return path_second_new,1


def one_route_exchange(problem,existing_solution,step):
    improved_solution=copy.deepcopy(existing_solution)
    paths=np.random.choice(existing_solution,2,replace=False)
    for path in paths:
        if len(path)==2:
            return existing_solution
    path_first=paths[0]
    path_second=paths[1]
    firstp=0
    secondp=0
    while(firstp%2==0):
        firstp=np.random.choice(path_first,1,replace=False)
    while(secondp%2==0):
        secondp=np.random.choice(path_second,1,replace=False)
    firstd=find_point_index_in_path(firstp+1,path_first)
    firstp=find_point_index_in_path(firstp,path_first)
    secondd=find_point_index_in_path(secondp+1,path_second)
    secondp=find_point_index_in_path(secondp,path_second)
    diff= problem.get_capacity(path_first[firstp]) - problem.get_capacity(path_second[secondp])
    if(diff==0):
        path_second_new=path_second[:secondp]+[path_first[firstp]]+path_second[secondp+1:secondd]+[path_first[firstd]]+path_second[secondd+1:]
        path_first_new = path_first[:firstp] + [path_second[secondp]] + path_first[firstp + 1:firstd] + [
            path_second[secondd]] + path_first[firstd + 1:]
    elif(diff>0):
        consumption = calculate_consumption(problem,path_second)
        if(consumption[secondp]+diff>problem.get_capacity(0)):
            return existing_solution
        path_first_new = path_first[:firstp] + [path_second[secondp]] + path_first[firstp + 1:firstd] + [
            path_second[secondd]] + path_first[firstd + 1:]
        for index in range(secondp+1,secondd+1):
            if(consumption[index]+diff>problem.get_capacity(0)):
                break
        path_second_new=path_second[:secondp]+[path_first[firstp]]+path_second[secondp+1:index]+[path_first[firstd]]+path_second[index:secondd]+path_second[secondd+1:]
    else:
        diff=0-diff
        consumption = calculate_consumption(problem, path_first)
        if (consumption[firstp] + diff > problem.get_capacity(0)):
            return existing_solution
        path_second_new = path_second[:secondp] + [path_first[firstp]] + path_second[(secondp + 1):secondd] + [
            path_first[firstd]] + path_second[(secondd + 1):]
        for index in range(firstp + 1, firstd + 1):
            if (consumption[index] + diff > problem.get_capacity(0)):
                break
        path_first_new = path_first[:firstp] + [path_second[secondp]] + path_first[(firstp + 1): index] + [
            path_second[secondd]] + path_first[index:firstd] + path_first[(firstd+1):]

    #if(is_illegal_path(problem,path_first_new)or is_illegal_path(problem,path_second_new)):
    #    print("nop")
    #    return existing_solution
    #print(path_first,'pf')
    #print(path_second,'ps')
    #print(path_first_new,'npf')
    #print(path_second_new,'nps')
    index_first=find_path_index_in_solution(path_first,existing_solution)
    index_second=find_path_index_in_solution(path_second,existing_solution)
    improved_solution[index_first]=copy.deepcopy(path_first_new)
    improved_solution[index_second]=copy.deepcopy(path_second_new)
    problem.mark_change_at(step,[index_second,index_first])
    return improved_solution


def reconstruct_solution_by_pair_exchange(problem, existing_solution, step, num_paths_to_ruin = config.num_paths_to_ruin,initializition = False):
    solution_to_return = copy.deepcopy(existing_solution)
    #print(existing_solution)
    path_to_ruin = np.random.choice(existing_solution,num_paths_to_ruin,replace=False)
    for path in path_to_ruin:
        if len(path)<=5:
            continue
        path_index = find_path_index_in_solution(path,solution_to_return)
        for i in range(3):
            firstp = 0
            secondp = 0
            while(firstp%2==0):
                firstp = np.random.choice(path,1,replace=False)
            while(secondp%2==0 or secondp==firstp) :
                secondp = np.random.choice(path,1,replace=False)
            secondd=find_point_index_in_path(secondp+1,path)
            firstd=find_point_index_in_path(firstp+1,path)
            secondp=find_point_index_in_path(secondp,path)
            firstp=find_point_index_in_path(firstp,path)
            path_new = do_exchange_path_pair(path, firstp, secondp, firstd, secondd)
            if(is_illegal_path(problem,path_new)):
                #print(path_new)
                #print("hyha")
                continue
            solution_to_return[path_index]=copy.deepcopy(path_new)
            problem.mark_change_at(step,[path_index])
            break
    #print(solution_to_return)
    return solution_to_return



def reconstruct_solution(problem, existing_solution, step, num_paths_to_ruin = config.num_paths_to_ruin,initializition = False):
    #print(existing_solution)
    #stt = time.time()
    succeed_all = 0
    try_times = 0
    while(succeed_all==0):
        candidate_indices = []
        path_num = num_paths_to_ruin
        customer_num = 0
        customers_to_reconstruct = []
        reconstructed_paths=[]
        capacities_left=[]
        for i in range(0,path_num):
            reconstructed_paths.append([0])
            capacities_left.append(problem.get_capacity(0))
        for path_index in range(len(existing_solution)):
            #if len(existing_solution[path_index]) > 2:
            #    candidate_indices.append(path_index)
            candidate_indices.append(path_index)
        paths_ruined = np.random.choice(candidate_indices, num_paths_to_ruin, replace=False)
        for path in paths_ruined:
            #print(existing_solution[path])
            #print(len(existing_solution[path]))
            customer_num += len(existing_solution[path]) -2
            customers_to_reconstruct.extend(existing_solution[path])
        for i in range(0,customer_num):
            #print(customer_num)
            succeed = 0
            while succeed==0:
                customer_in = np.random.choice(customers_to_reconstruct)
                if customer_in%2 ==0 and (customer_in-1) in customers_to_reconstruct:
                    continue
                if customer_in%2 == 1 and problem.get_capacity(customer_in) > np.max(capacities_left):
                    continue
                can_paths= get_can_path(problem,reconstructed_paths,capacities_left,customer_in)
                if(len(can_paths)==0):
                    continue
                path_in = np.random.choice(range(0,len(can_paths)))
                path_in = can_paths[path_in]
                for i in range(0,len(reconstructed_paths)):
                    if path_in == reconstructed_paths[i]:
                        path_in_index = i
                
                reconstructed_paths[path_in_index].append(customer_in)
                customers_to_reconstruct.remove(customer_in)
                capacities_left[path_in_index]-=problem.get_capacity(customer_in)
                #print(customer_in)
                #print(reconstructed_paths)
                #print(capacities_left)
                succeed = 1

        succeed_all = 1
        for path in reconstructed_paths:
            path.append(0)
            if is_illegal_path(problem,path) and not initializition:
                succeed_all = 0
                #if initializition:
                #    print(calculate_path_distance(problem,path))
                #print("yab")
                break
            elif initializition:
                #print(path)
                if calculate_path_distance(problem,path)>1.4*config.max_path_distance:
                    succeed_all = 0
                    break
        #print("r")
        #print(time.time()-stt)
        path_slice=[]
        for index in paths_ruined:
            path_slice.append(existing_solution[index])
        if calculate_solution_distance(problem,reconstructed_paths) == calculate_solution_distance(problem,path_slice):
            succeed_all = 0
        if try_times>100 and not initializition:
            return existing_solution
    problem.mark_change_at(step, paths_ruined)
    improved_solution = copy.deepcopy(existing_solution)
    for i in range(0,len(reconstructed_paths)):
        improved_solution[paths_ruined[i]] = copy.deepcopy(reconstructed_paths[i])
    #print(improved_solution,'ipv')
    #print("all")
    return improved_solution



def get_can_path(problem, reconstructed_path, capacities_left,customer_in):
    can_paths = []
    if customer_in%2 ==1:
        for i in range(0,len(capacities_left)):
            if capacities_left[i]>=problem.get_capacity(customer_in):
                can_paths.append(reconstructed_path[i])
        return can_paths
    else:
        for i in range(0,len(capacities_left)):
            if (customer_in-1) in reconstructed_path[i]:
                can_paths.append(reconstructed_path[i])
                break
        return can_paths


def reconstruct_solution_dep(problem, existing_solution, step):
    distance_hash = round(calculate_solution_distance(problem, existing_solution) * 1e6)
    if config.detect_negative_cycle and distance_hash not in problem.distance_hashes:
        problem.add_distance_hash(distance_hash)
        positive_cycles = []
        cycle_selected = None
        for capacity in range(1, 10):
            # TODO: relax the requirement of ==capacity
            # TODO: caching, sparsify
            graph = construct_graph(problem, existing_solution, capacity)
            negative_cycle, flag = graph.find_negative_cycle()
            if negative_cycle:
                if flag == -1.0:
                    cycle_selected = negative_cycle
                    break
                else:
                    positive_cycles.append(negative_cycle)
        if cycle_selected is None and len(positive_cycles) > 0:
            index = np.random.choice(range(len(positive_cycles)), 1)[0]
            cycle_selected = positive_cycles[index]
        if cycle_selected is not None:
                negative_cycle = cycle_selected
                improved_solution = copy.deepcopy(existing_solution)
                customers = []
                for pair in negative_cycle:
                    path_index, node_index = pair[0], pair[1]
                    customers.append(improved_solution[path_index][node_index])
                customers = [customers[-1]] + customers[:-1]
                for index in range(len(negative_cycle)):
                    pair = negative_cycle[index]
                    path_index, node_index = pair[0], pair[1]
                    improved_solution[path_index][node_index] = customers[index]
                    problem.mark_change_at(step, [path_index])
                # if not validate_solution(problem, improved_solution):
                #     print('existing_solution={}, invalid improved_solution={}, negative_cycle={}'.format(
                #         existing_solution, improved_solution, negative_cycle))
                # else:
                #     print('cost={}, negative_cycle={}'.format(
                #         calculate_solution_distance(problem, improved_solution) - calculate_solution_distance(problem, existing_solution),
                #         negative_cycle)
                #     )
                return improved_solution

    solution = []
    n = problem.get_num_customers()
    customer_indices = list(range(n + 1))

    candidate_indices = []
    for path_index in range(len(existing_solution)):
        if len(existing_solution[path_index]) > 2:
            candidate_indices.append(path_index)
    paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
    start_customer_index = n + 1
    for path_index in paths_ruined:
        path = existing_solution[path_index]
        for customer_index in path:
            if customer_index == 0:
                continue
            start_customer_index -= 1
            customer_indices[start_customer_index] = customer_index

    if np.random.uniform() < 0.5:
        while len(solution) == 0:
            paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
            solution = reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined)
    else:
        trip = [0]
        capacity_left = problem.get_capacity(0)
        i = start_customer_index
        while i <= n:
            to_indices = []
            adjusted_distances = []
            # if len(trip) > 1:
            #     to_indices.append(0)
            #     adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], 0))
            p=0
            for j in range(i, n + 1):
                if problem.get_capacity(customer_indices[j]) > capacity_left:
                    continue
                
                if customer_indices[j] > 0 and customer_indices[j] % 2 == 0 and ((customer_indices[j] - 1) in customer_indices[start_customer_index:n+1] and (customer_indices[j] - 1) not in customer_indices[start_customer_index:i]):
                    print(customer_indices[start_customer_index:i])
                    
                    print(customer_indices)
                    print(customer_indices[j])
                    print("j="+str(j))
                    continue
                to_indices.append(j)
                adjusted_distances.append(
                    calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
            random_index = sample_next_index(to_indices, adjusted_distances)

            if random_index == 0 :
                trip.append(0)
                solution.append(trip)
                trip = [0]
                capacity_left = problem.get_capacity(0)
                continue
            customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
            trip.append(customer_indices[i])
            capacity_left -= problem.get_capacity(customer_indices[i])
            i += 1
        if len(trip) > 1:
            trip.append(0)
            solution.append(trip)

    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    while len(solution) < len(paths_ruined):
        solution.append([0, 0])
    improved_solution = copy.deepcopy(existing_solution)
    solution_index = 0
    for path_index in sorted(paths_ruined):
        improved_solution[path_index] = copy.deepcopy(solution[solution_index])
        solution_index += 1
    problem.mark_change_at(step, paths_ruined)
    for solution_index in range(len(paths_ruined), len(solution)):
        improved_solution.append(copy.deepcopy(solution[solution_index]))
        problem.mark_change_at(step, [len(improved_solution) - 1])

    has_seen_empty_path = False
    for path_index in range(len(improved_solution)):
        if len(improved_solution[path_index]) == 2:
            if has_seen_empty_path:
                empty_slot_index = path_index
                for next_path_index in range(path_index + 1, len(improved_solution)):
                    if len(improved_solution[next_path_index]) > 2:
                        improved_solution[empty_slot_index] = copy.deepcopy(improved_solution[next_path_index])
                        empty_slot_index += 1
                improved_solution = improved_solution[:empty_slot_index]
                problem.mark_change_at(step, range(path_index, empty_slot_index))
                break
            else:
                has_seen_empty_path = True
    return improved_solution


def get_num_points(config):
    if config.model_to_restore is None:
        return config.num_training_points
    else:
        return config.num_test_points


def generate_problem():
    np.random.seed(config.problem_seed)
    random.seed(config.problem_seed)
    config.problem_seed += 1

    num_sample_points = get_num_points(config)
    if config.problem == 'vrp'and not config.baseline_exp:
        num_sample_points += 1
    locations = np.random.uniform(size=(num_sample_points, 2))
    if config.problem == 'vrp' and not config.baseline_exp:
        if config.depot_positioning == 'C':
            locations[0][0] = 0.5
            locations[0][1] = 0.5
        elif config.depot_positioning == 'E':
            locations[0][0] = 0.0
            locations[0][1] = 0.0
        if config.customer_positioning in {'C', 'RC'}:
            S = np.random.randint(6) + 3
            centers = locations[1 : (S + 1)]
            grid_centers, probabilities = [], []
            for x in range(0, 1000):
                for y in range(0, 1000):
                    grid_center = [(x + 0.5) / 1000.0, (y + 0.5) / 1000.0]
                    p = 0.0
                    for center in centers:
                        distance = calculate_distance(grid_center, center)
                        p += math.exp(-distance * 1000.0 / 40.0)
                    grid_centers.append(grid_center)
                    probabilities.append(p)
            probabilities = np.asarray(probabilities) / np.sum(probabilities)
            if config.customer_positioning in 'C':
                num_clustered_locations = get_num_points(config) - S
            else:
                num_clustered_locations = get_num_points(config) // 2 - S
            grid_indices = np.random.choice(range(len(grid_centers)), num_clustered_locations, p=probabilities)
            for index in range(num_clustered_locations):
                grid_index = grid_indices[index]
                locations[index + S + 1][0] = grid_centers[grid_index][0] + (np.random.uniform() - 0.5) / 1000.0
                locations[index + S + 1][1] = grid_centers[grid_index][1] + (np.random.uniform() - 0.5) / 1000.0
    if config.baseline_exp:
        #locations=[[0,0],[1,0],[0,1],[0,-1],[-1,0],[1,1],[2,1],[3,1],[-1,1],[-2,-1],[-2,2],
        #           [-3,-1],[-1,2],[2,3],[2,1],[-1,-2],[0,2]]
        '''locations=[[0,0],
                   [1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0],[9,0],[10,0],
                   [1,1],[2,1],[3,2],[4,3],[5,5],[6,6],[7,2],[8,3],[9,4],[10,2],
                   [2,1],[3,1],[4,2],[1,3],[5,5],[7,6],[2,2],[4,3],[8,4],[0,2],
                   [2,8],[2,1],[4,12],[12,3],[6,5],[7,7],[3,2],[3,3],[6,4],[1,2],
                   [2,1],[1,1],[0,2],[1,3],[5,5],[7,6],[2,0],[0,3],[2,4],[0,6]]
'''
        locations=[[0,0],[-9, 17], [-16, -50], [19, -26], [28, 8], [12, 14], [-45, -5], [31, -23], [11, 41], [45, -8], [-23, -14], [41, -46], [-48, 3], [42, 32], [-29, -34], [-32, 45], [-3, -24], [21, -12], [19, -38], [17, 49], [-15, 44], [-47, -39], [-28, -17], [23, 14], [-9, -39], [3, 18], [-3, -6], [12, 7], [-13, 9], [-27, -9], [-21, 28], [-34, -15], [40, -8], [38, -44], [-10, -8], [14, -2], [-4, -45], [40, -21], [20, 0], [-44, -49], [43, -2], [-21, -27], [34, 4], [6, -10], [16, 26], [-19, -42], [-6, -11], [-24, -27], [-13, -12], [-32, 32], [-21, -9]]
        '''
        locations=[[0 for i in range(2)]for j in range(num_sample_points+1)]
        for i in range(1,num_sample_points+1):
            locations[i][0]=np.random.randint(-50,50)
            locations[i][1] = np.random.randint(-50, 50)
        '''
    capacities = get_random_capacities(len(locations))
    #print(capacities)
    TWs = get_time_windows(len(locations))
    #print(TWs)
    problem = Problem(locations, capacities,TWs)
    np.random.seed(config.problem_seed * 10)
    random.seed(config.problem_seed * 10)
    return problem


ATTENTION_ROLLOUT, LSTM_ROLLOUT = False, True


def embedding_net_nothing(input_):
    return input_


def embedding_net_2(input_):
    with tf.variable_scope("embedding_net"):
        architecture_type = 0
        if architecture_type == 0:
            print(input_.shape,'imp')
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True,
                          BN=True, initializer=tf.contrib.layers.xavier_initializer())

            layer_attention = encode_seq(input_seq=x, input_dim=config.num_embedded_dim_1, num_stacks=1, num_heads=8,
                                         num_neurons=64, is_training=True, dropout_rate=0.1)
            print(layer_attention.shape,'la')
            # layer_attention = tf.reshape(layer_attention, [-1, (config.num_training_points) * config.num_embedded_dim_1])
            # layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim_2, activation_fn=tf.nn.relu)
            # layer_2 = tf.nn.dropout(layer_2, keep_prob)
            layer_2 = tf.reduce_sum(layer_attention, axis=1)
            print(layer_2.shape,'l2')
        else:
            #TODO:
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True, BN=False, initializer=tf.contrib.layers.xavier_initializer())
            x = tf.reduce_sum(x, axis=1)
            layer_2 = tf.nn.relu(x)
    return layer_2


def embedding_net(input_):
    #tf.reset_default_graph()
    with tf.variable_scope("embedding_net"):
        first_trip = input_[0]
        first_trip = tf.reshape(first_trip, [-1, config.max_points_per_trip, config.input_embedded_trip_dim])
        trip_embedding = embedding_net_lstm(first_trip)
        for trip_index in range(1, input_.shape[0]):
            with tf.variable_scope("lstm", reuse=True):
                trip = input_[trip_index]
                trip = tf.reshape(trip, [-1, config.max_points_per_trip, config.input_embedded_trip_dim])
                current_trip_embedding = embedding_net_lstm(trip)
                trip_embedding = tf.concat([trip_embedding, current_trip_embedding], axis=1)
        attention_embedding = embedding_net_attention(trip_embedding)
    return attention_embedding


def embed_trip(trip, points_in_trip):
    trip_prev = np.vstack((trip[-1], trip[:-1]))
    trip_next = np.vstack((trip[1:], trip[0]))
    distance_from_prev = np.reshape(np.linalg.norm(trip_prev - trip, axis=1), (points_in_trip, 1))
    distance_to_next = np.reshape(np.linalg.norm(trip - trip_next, axis=1), (points_in_trip, 1))
    distance_from_to_next = np.reshape(np.linalg.norm(trip_prev - trip_next, axis=1), (points_in_trip, 1))
    trip_with_additional_information = np.hstack((trip_prev, trip, trip_next, distance_from_prev, distance_to_next, distance_from_to_next))
    return trip_with_additional_information


def embedding_net_lstm(input_):
    #print(input_)
    #print(input_.shape,'lstminput')
    seq = tf.unstack(input_, input_.shape[1], 1)
    #print(seq,'seq')
    num_hidden = 128
    with tf.variable_scope("lstm_embeding", reuse=tf.AUTO_REUSE):
        lstm_fw_cell1 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell1 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell1, lstm_bw_cell1, seq, dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell1, lstm_bw_cell1, seq, dtype=tf.float32)
        layer_lstm = outputs[-1]
        layer_2 = tf.contrib.layers.fully_connected(layer_lstm, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_2 = tf.reshape(layer_2, [-1, 1, config.num_embedded_dim])
    return layer_2


def embedding_net_attention(input_):
    #print(input_.shape)
    #tf.reset_default_graph()
    with tf.variable_scope("attention_embedding",reuse=tf.AUTO_REUSE):
        x = p_embed_seq(input_seq=input_, from_=config.num_embedded_dim, to_=128, is_training=True, BN=True, initializer=tf.contrib.layers.xavier_initializer())
        layer_attention = p_encode_seq(input_seq=x, input_dim=128, num_stacks=3, num_heads=16, num_neurons=512, is_training=True, dropout_rate=0.1)
        layer_attention = tf.reshape(layer_attention, [-1, input_.shape[0] * 128])
        layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
    return layer_2


def embed_solution(problem, solution):
    embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
    n_trip = len(solution)
    for trip_index in range(min(config.max_trips_per_solution, n_trip)):
        trip = solution[trip_index]
        truncated_trip_length = np.minimum(config.max_points_per_trip, len(trip) - 1)
        if truncated_trip_length > 1:
            points_with_coordinate = np.zeros((truncated_trip_length, 2))
            for point_index in range(truncated_trip_length):
                points_with_coordinate[point_index] = problem.get_location(trip[point_index])
            embedded_solution[trip_index, :truncated_trip_length] = copy.deepcopy(embed_trip(points_with_coordinate, truncated_trip_length))
    return embedded_solution


def embed_solution_with_nothing(problem, solution):
    embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
    return embedded_solution


def embed_path_with_nodes(problem,path,node_embedding=None):
    #path = path.tolist()
    if len(path)==2:
        embedded_node_in_this_path = [[0]*config.input_embedded_trip_dim_2]*config.max_points_per_trip
        return embedded_node_in_this_path

    embedded_node_in_this_path = []
    n = len(path)-1
    if node_embedding is not None:
        for index in range(1,n):
            customer = path[index]
            embedded_node_in_this_path.append(node_embedding[customer-1])
    else:
        consumption = calculate_consumption(problem,path)
        time_series = calculate_time_series(problem,path)
        embedded_node_in_this_path = []
        for index in range(1,n):
            customer = path[index]
            embedded_input = []
            embedded_input.append(problem.get_capacity(customer))
            embedded_input.extend(problem.get_location(customer))
            embedded_input.append(problem.get_capacity(0) - max(consumption))
            embedded_input.extend(problem.get_location(path[index - 1]))
            embedded_input.extend(problem.get_location(path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], customer))
            embedded_input.append(problem.get_distance(customer, path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], path[index + 1]))
            embedded_input.append(problem.get_TW(path[index],0))
            embedded_input.append(problem.get_TW(path[index],1))
            embedded_input.append(time_series[index][0])
            embedded_input.append(time_series[index][1])
            embedded_node_in_this_path.append(embedded_input)
    if(len(embedded_node_in_this_path)<config.max_points_per_trip):
        diff = config.max_points_per_trip-len(embedded_node_in_this_path)
        embedded_node_in_this_path = embedded_node_in_this_path+[[0]*15]*diff
    #print(embedded_node_in_this_path,'enitp')
    #print(len(embedded_node_in_this_path))
    return embedded_node_in_this_path


def embed_solution_with_attention(problem, solution):
    embedded_solution = np.zeros((config.num_training_points, config.input_embedded_trip_dim_2))

    for path in solution:
        if len(path) == 2:
            continue
        n = len(path) - 1
        consumption = calculate_consumption(problem, path)
        time_series = calculate_time_series(problem,path)
        for index in range(1, n):
            customer = path[index]
            embedded_input = []
            embedded_input.append(problem.get_capacity(customer))
            embedded_input.extend(problem.get_location(customer))
            embedded_input.append(problem.get_capacity(0) - max(consumption))
            embedded_input.extend(problem.get_location(path[index - 1]))
            embedded_input.extend(problem.get_location(path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], customer))
            embedded_input.append(problem.get_distance(customer, path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], path[index + 1]))
            embedded_input.append(problem.get_TW(path[index],0))
            embedded_input.append(problem.get_TW(path[index],1))
            embedded_input.append(time_series[index][0])
            embedded_input.append(time_series[index][1])
            for embedded_input_index in range(len(embedded_input)):
                embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
    #print("a")
    #print(embedded_solution.shape)
    return embedded_solution


TEST_X = tf.placeholder(tf.float32, [None, config.num_training_points, config.input_embedded_trip_dim_2])
embedded_x = embedding_net_2(TEST_X)
PATH_ALL = tf.placeholder(tf.float32, [config.num_paths, config.max_points_per_trip,config.input_embedded_trip_dim_2])
embedded_path_all = embedding_net(PATH_ALL)
LAST_A1 =  tf.placeholder(tf.float32,[1, config.max_points_per_trip,config.input_embedded_trip_dim_2])
LAST_A2 = tf.placeholder(tf.float32,[1, config.max_points_per_trip,config.input_embedded_trip_dim_2]) 
embedded_path_all = embedding_net(PATH_ALL)
embedded_a1 = embedding_net(LAST_A1)
embedded_a2 = embedding_net(LAST_A2)
env_observation_space_n = config.num_history_action_use * 2 + 5
action_labels_placeholder = tf.placeholder("float", [None, config.num_actions - 1])
#print(embedded_path_all.shape,'pashape')
#print(tf.reduce_mean(embedded_path_all,axis=0,keep_dims=True).shape,'meanshape')
#full_states_target = tf.concat([tf.reduce_mean(embedded_path_all,axis=0,keepdims=True),embedded_a1,embedded_a2],axis=1)
#print(full_states_target.shape)


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=config.policy_learning_rate, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.float32, [None, env_observation_space_n], "states")
            if config.use_attention_embedding:
                full_states = tf.concat([self.states, embedded_x], axis=1)
            else:
                full_states = self.states

            self.hidden1 = tf.contrib.layers.fully_connected(
                inputs=full_states,
                num_outputs=config.hidden_layer_dim,
                activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.hidden1,
                num_outputs=config.num_actions - 1,
                activation_fn=None)

            self.action_probs = tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.0)
            log_prob = tf.log(self.action_probs)
            self.action_choose = tf.multinomial(log_prob,1)
            action_one_hot = tf.squeeze(tf.one_hot(self.action_choose,depth=config.num_actions-1),[0])
            
            ##################################################
            full_states_target = tf.concat([tf.reduce_mean(embedded_path_all,axis=0,keepdims=True),embedded_a1,embedded_a2,self.states,action_one_hot],axis=1)
            num_units = config.num_embedded_dim_1
            C = 10
            self.Q = tf.layers.dense(inputs=full_states_target, units=num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
            self.K = tf.layers.dense(inputs=embedded_path_all, units=num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
	    #self.V = tf.layers.dense(embedded_path_all, num_units, activation=tf.nn.relu)  # [batch_size, seq_length, n_hidden]
	    # Multiplication
            outputs = tf.matmul(self.Q, tf.transpose(self.K, [1, 0]))  # num_heads*[batch_size, seq_length, seq_length]
	    # Scale
            outputs = outputs / (self.K.get_shape().as_list()[-1] ** 0.5)
	    # Activation
            log_path_probs = C * tf.nn.tanh(outputs)
            #self.path_probs = tf.clip_by_value(tf.nn.softmax(outputs),1e-10,1.0)  # num_heads*[batch_size, seq_length, seq_length]
            self.path_probs = tf.clip_by_value(tf.nn.softmax(log_path_probs),1e-10,1.0)
            ###################################################
            

            #https://stackoverflow.com/questions/33712178/tensorflow-nan-bug?newreg=c7e31a867765444280ba3ca50b657a07
            # training part of graph
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

            '''
            self.path_aims_1=tf.placeholder(tf.int32,[None],name="path_aims_1")
            self.path_aims_2=tf.placeholder(tf.int32,[None],name="path_aims_2")
            log_path_prob_1 = tf.log(self.path_probs)
            #log_path_prob_2 = tf.log(self.path_aims_2)
            indices_path_1 = tf.range(0, tf.shape(log_path_prob_1)[0]) * tf.shape(log_path_prob_1)[1] + self.path_aims_1
            indices_path_2 = tf.range(0, tf.shape(log_path_prob_1)[0]) * tf.shape(log_path_prob_1)[1] + self.path_aims_2
            path_prob_1 = tf.gather(tf.reshape(log_path_prob_1, [-1]), indices_path_1)
            path_prob_2 = tf.gather(tf.reshape(log_path_prob_1, [-1]), indices_path_2)
            '''
            self.path_probs_1=tf.placeholder(tf.float32,[None],name="path_probs_1")
            self.path_probs_2=tf.placeholder(tf.float32,[None],name="path_probs_2")
            path_prob_1 = tf.log(self.path_probs_1)
            path_prob_2 = tf.log(self.path_probs_2)
            self.loss = -tf.reduce_sum(act_prob * path_prob_1 * path_prob_2 * self.advantages)
            # self.loss = -tf.reduce_mean(act_prob * self.advantages)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            # Define loss and optimizer
            self.sl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=action_labels_placeholder))
            self.sl_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.sl_train_op = self.sl_optimizer.minimize(self.sl_loss)
            # Training accuracy
            correct_pred = tf.equal(tf.argmax(self.action_probs, 1), tf.argmax(action_labels_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def predict(self, states, test_x, path_all,last_a1,last_a2,sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, TEST_X: test_x,PATH_ALL:path_all,LAST_A1:last_a1,LAST_A2:last_a2, keep_prob: 1.0}
        else:
            feed_dict = {self.states: states, keep_prob:1.0}

        return sess.run([self.action_choose,self.path_probs],feed_dict)

    def update(self, states, test_x, advantages, actions, path_aims_1,path_aims_2,sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, self.path_probs_1:path_aims_1,self.path_probs_2:path_aims_2,TEST_X:test_x, keep_prob:1.0}
        else:
            feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, self.path_probs_1:path_aims_1,self.path_probs_2:path_aims_2,keep_prob: 1.0}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def train(self, states, test_x, action_labels, sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, TEST_X:test_x, action_labels_placeholder: action_labels, keep_prob:1.0}
        else:
            feed_dict = {self.states: states, action_labels_placeholder: action_labels, keep_prob:1.0}
        _, loss, accuracy = sess.run([self.sl_train_op, self.sl_loss, self.accuracy], feed_dict)
        return loss, accuracy


previous_solution = None
initial_solution = None
best_solution = None


def env_act(step, problem, min_distance, solution, distance, action,path_aim_1,path_aim_2):
    global initial_solution
    global previous_solution
    #print(step)
    #print(min_distance,'mdist')
    #print(distance,'dst')
    #print(calculate_solution_distance(problem,solution),'ccdst')
    #print(action,'act'mprove)
    if action > 0:
        next_solution, delta = improve_solution_by_action(step, problem, solution, action,path_aim_1,path_aim_2)
        next_distance = distance + delta
        #if(abs(next_distance-calculate_solution_distance(problem,next_solution))>=1):
        #    print(next_distance,'nxtdst')
        #    print(calculate_solution_distance(problem,next_solution),'nxtccdst')
        if config.debug_mode:
            if not validate_solution(problem, next_solution, next_distance):
                print('Invalid solution!')
    else:
        problem.record_solution(solution, distance)
        if distance / min_distance < 1.01:
            previous_solution = solution
            next_solution = construct_solution(problem, solution, step)
        else:
            previous_solution = best_solution
            next_solution = construct_solution(problem, best_solution, step)
            # problem.reset_change_at_and_no_improvement_at()
        next_distance = calculate_solution_distance(problem, next_solution)
        initial_solution = next_solution
        #print(next_solution)
    #print(step)
    #print(action,'act')
    #calculate_time_cost(problem,next_solution[0])
    #if not validate_solution(problem,next_solution,next_distance):
    #    print(solution,'debug')
    #    print(next_solution)
    return next_solution, next_distance


action_timers = [0.0] * (config.num_actions * 2)


def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None):
    if state is None or action == 0:
        next_state = [0.0, 0.0, 0]
        for _ in range(config.num_history_action_use):
            next_state.append(0.0)
            next_state.append(0)
        next_state.append(0.0)
        next_state.append(0)
    else:
        delta = next_distance - distance
        if delta < -EPSILON:
            delta_sign = -1.0
        else:
            delta_sign = 1.0
        next_state = [0.0, next_distance - min_distance, delta]
        if config.num_history_action_use != 0:
            next_state.extend(state[(-config.num_history_action_use * 2):])
        next_state.append(delta_sign)
        next_state.append(action)
    return next_state


def env_step(step, state, problem, min_distance, solution, distance, action, path_aim_1,path_aim_2,record_time=True):
    start_timer = datetime.datetime.now()
    #stt=time.time()
    next_trip, next_distance = env_act(step, problem, min_distance, solution, distance, action,path_aim_1,path_aim_2)

    next_state = env_generate_state(min_distance, state, action, distance, next_distance)
    reward = distance - next_distance
    end_timer = datetime.datetime.now()
    if record_time:
        action_timers[action * 2] += 1
        action_timers[action * 2 + 1] += (end_timer - start_timer).total_seconds()
    done = (datetime.datetime.now() - env_start_time).total_seconds() >= config.max_rollout_seconds
    #print(next_trip)
    #print(time.time()-stt)
    #three_route_exchange(problem,solution,step)
    return next_state, reward, done, next_trip, next_distance


def format_print(value):
    return round(float(value), 2)


def format_print_array(values):
    results = []
    for value in values:
        results.append(format_print(value))
    return results


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print ([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def should_restart(min_distance, distance, no_improvement):
    if no_improvement >= config.max_no_improvement:
        return True
    if no_improvement <= config.max_no_improvement - 1:
        return False
    percentage_over = round((distance / min_distance - 1.0) * 100)
    upper_limits = [20, 10, 5, 2]
    return percentage_over >= upper_limits[no_improvement - 2]


def get_edge_set(solution):
    edge_set = set()
    for path in solution:
        if len(path) > 2:
            for path_index in range(1, len(path)):
                node_before = path[path_index - 1]
                node_current = path[path_index]
                value = '{}_{}'.format(min(node_before, node_current), max(node_before, node_current))
                edge_set.add(value)
    return edge_set


def calculate_solution_similarity(solutions):
    edge_set = get_edge_set(solutions[0])
    for solution in solutions[1:]:
        edge_set = edge_set.intersection(get_edge_set(solution))
    return len(edge_set)


def sort_solution(problem,solution):
    sorted_solution=[solution[0]]
    costs = [calculate_path_distance(problem,solution[0])]
    for i in range(1,len(solution)):
        cost = calculate_path_distance(problem,solution[i])
        flag = 0
        for j in range(len(costs)):
            if cost>costs[j]:
                sorted_solution=sorted_solution[:j]+[solution[i]]+sorted_solution[j:]
                costs=costs[:j]+[cost]+costs[j:]
                flag = 1
                break
        if not flag:
            sorted_solution.append(solution[i])
            costs.append(cost)
    #print(costs)
    return sorted_solution


def embed_solution_with_path(problem,solution,node_embedding=None):
    n = len(solution)
    embedded_path_all=[]
    for path in solution:
        embedded_path_all.append(embed_path_with_nodes(problem,path,node_embedding))
    return embedded_path_all


def takeSecond(elem):
    return elem[1]


def load_problem(index_sample):
    #index_sample = 2
    data = scio.loadmat('bridge_100.mat')
    locs=data['locs'][index_sample]
    caps=data['caps'][index_sample]
    TWs=data['TWs'][index_sample]
    #print(locations)
    #print(capacities)
    problem=Problem(locs,caps,TWs)
    return problem


def save_solution_bef(problem, solution):
    locs=[[] for i in range(len(solution))]
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            locs[i].append(problem.get_location(solution[i][j]))
    scio.savemat('solution_visible_bef_50.mat',{'sol':solution,'locs':locs})



def save_solution_aft(problem, solution):
    locs=[[] for i in range(len(solution))]
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            locs[i].append(problem.get_location(solution[i][j]))
    scio.savemat('solution_visible_aft_50.mat',{'sol':solution,'locs':locs})




gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    policy_estimator = PolicyEstimator()
    initialize_uninitialized(sess)
    print(sess.run(tf.report_uninitialized_variables()))
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable={}, Shape={}".format(k, v.shape))
    sys.stdout.flush()
    saver = tf.train.Saver()
    if config.model_to_restore is not None:
        saver.restore(sess, config.model_to_restore)

    distances = []
    steps = []
    consolidated_distances, consolidated_steps = [], []
    timers = []
    num_checkpoint = int(config.max_rollout_steps/config.step_interval)
    step_record = np.zeros((2, num_checkpoint))
    distance_record = np.zeros((2, num_checkpoint))
    start = datetime.datetime.now()
    seed = config.problem_seed
    tf.set_random_seed(seed)

    Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "path_aim_1","path_aim_2","reward", "next_state", "done"])
    main_steps=[]
    main_distances=[]
    main_percentages=[]
    main_actions=[]
    problem_1=generate_problem()
    problem_2=generate_problem()
    problem_3=generate_problem()
    main_steps_problem=[]
    main_distances_problem=[]
    main_actions_problem=[]
    R_anals_actions=[]
    R_anals_path_aims_1=[]
    R_anals_path_aims_2=[]
    Ranks_main=[]
    Prob_ranks_main=[]
    test_distances=[0]*8
    solutions=[]
    for index_sample in range(config.num_episode):
        states = []
        trips = []
        actions = []
        path_aims_1 = []
        path_aims_2 = []
        advantages = []
        action_labels = []
        stepss = []
        min_distances=[]
        Ranks=[0]*config.num_paths
        Prob_ranks=[]
        actions_record=[0]*(config.num_actions+1)
        if index_sample > 0 and index_sample % config.debug_steps == 0:
            if not config.use_random_rollout:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                count_timers = formatted_timers[4:][::2]
                time_timers = formatted_timers[4:][1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in count_timers])))
                start_active = ((len(distances) // config.num_active_learning_iterations) * config.num_active_learning_iterations)
                if start_active == len(distances):
                    start_active -= config.num_active_learning_iterations
                tail_distances = distances[start_active:]
                tail_steps = steps[start_active:]
                min_index = np.argmin(tail_distances)
                if config.num_active_learning_iterations == 1 or len(distances) % config.num_active_learning_iterations == 1:
                    consolidated_distances.append(tail_distances[min_index])
                    consolidated_steps.append(tail_steps[min_index] + min_index * config.max_rollout_steps)
                else:
                    consolidated_distances[-1] = tail_distances[min_index]
                    consolidated_steps[-1] = tail_steps[min_index] + min_index * config.max_rollout_steps
                print('index_sample={}, mean_distance={}, mean_step={}, tail_distance={}, last_distance={}, last_step={}, timers={}'.format(
                    index_sample,
                    format_print(np.mean(consolidated_distances)), format_print(np.mean(consolidated_steps)),
                    format_print(np.mean(consolidated_distances[max(0, len(consolidated_distances) - 1000):])),
                    format_print(consolidated_distances[-1]), consolidated_steps[-1],
                    formatted_timers[:4]
                ))
                sys.stdout.flush()
            else:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                for index in range(num_checkpoint):
                    print('rollout_num={}, index_sample={}, mean_distance={}, mean_step={}, last_distance={}, last_step={}, timers={}'.format(
                        (index + 1) * config.step_interval, index_sample, ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample,
                        ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample, distance_record[1, index],
                        step_record[1, index], formatted_timers[:4]
                    ))
                    step_record[0, index] = ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample
                    distance_record[0, index] = ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample
                sys.stdout.flush()

        if(config.with_test_set):
            if index_sample%8<=4:
                problem = generate_problem()
            elif index_sample%8==5:
                problem=problem_1
            elif index_sample%8==6:
                problem=problem_2
            else:
                problem=problem_3
        else:
            #problem = load_problem(index_sample)
            problem = generate_problem()
        #print(problem.get_capacity(10))
        
        #print(calculate_solution_distance(problem,solution))
        '''
        solutionst=[]
        solutionst.append([[[0, 0], [0, 0], [0, 5, 25, 6, 21, 31, 22, 47, 48, 26, 32, 37, 7, 38, 27, 28, 8, 19, 20, 15, 16, 17, 11, 18, 29, 12, 49, 30, 1, 50, 45, 2, 33, 46, 34, 0], [0, 43, 44, 23, 35, 24, 36, 3, 4,
13, 14, 39, 41, 42, 40, 9, 10, 0], [0, 0]]])
        solutionst.append([[[0, 3, 4, 37, 38, 9, 10, 27, 28, 15, 16, 39, 25, 40, 26, 0], [0, 35, 7, 33, 36, 29, 34, 43, 44, 8, 19, 20, 49, 30, 23, 50, 45, 24, 46, 0], [0, 13, 14, 0], [0, 0], [0, 17, 5, 6, 47, 18, 11, 48, 1, 12, 31, 21, 2, 41, 32, 42, 22, 0]]])
        solutionst.append([[[0, 5, 35, 29, 36, 17, 6, 18, 11, 49, 30, 50, 12, 15, 16, 0], [0, 43, 7, 47, 44, 19, 8, 20, 48, 0], [0, 37, 38, 13,
14, 0], [0, 21, 31, 41, 32, 42, 22, 0], [0, 3, 4, 33, 34, 9, 10, 27, 28, 1, 23, 24, 45, 25, 46, 2, 39, 40, 26, 0]]])
        solutionst.append([[[0, 17, 5, 7, 6, 47, 18, 11, 48, 8, 1, 45, 12, 25, 46, 31, 2, 41, 26, 21, 32, 42, 22, 0], [0, 3, 4, 37, 38, 13, 14, 0], [0, 43, 29, 44, 19, 30, 49, 20, 23,
50, 24, 0], [0, 0], [0, 35, 33, 36, 34, 9, 10, 27, 28, 15, 16, 39, 40, 0]]] )
        solutionst.append([[[0, 3, 4, 37, 38, 13, 14, 0], [0, 0], [0, 43, 29, 49, 44, 19, 30, 20, 23, 50, 24, 45, 1, 31,
25, 2, 46, 41, 26, 21, 42, 32, 22, 0], [0, 35, 33, 36, 34, 9, 10, 27, 28, 0], [0, 17, 5, 7, 6,
47, 18, 11, 8, 48, 12, 15, 16, 39, 40, 0]]])
        solutionst.append([[[0, 43, 7, 47, 44, 19, 20, 8, 23, 24, 48, 0], [0, 3, 4, 33, 34, 9, 10, 0], [0, 11, 49, 50, 1, 45, 12, 46, 39, 2, 40, 0], [0, 37, 38, 13, 14, 0], [0, 5, 35, 29, 36, 17, 6, 18, 27, 30, 28, 15, 16, 25, 41, 31, 26, 21, 42, 32, 22, 0]]])
        for st in solutionst:
            solution=st[0]
            print(solution,'sol')
            print(calculate_solution_distance_dep(problem,solution))
            calculate_outside_ratio(problem,solution)

        '''


        solution = construct_solution(problem)
        print(calculate_solution_distance(problem,solution))
        #solution = sort_solution(problem, solution)
        print(solution,'std')
        solution_before = copy.deepcopy(solution)
        #save_solution_bef(problem,solution)
        #save_solution_aft(problem,solution)
        best_solution = copy.deepcopy(solution)

        if config.use_attention_embedding:
            embedded_trip = embed_solution_with_attention(problem, solution)
        else:
            embedded_trip = [0]
        last_path_aim_1=[[[0]*config.input_embedded_trip_dim_2]*config.max_points_per_trip]
        last_path_aim_2=[[[0]*config.input_embedded_trip_dim_2]*config.max_points_per_trip]
        embedded_paths_all = embed_solution_with_path(problem,solution)

        min_distance = calculate_solution_distance(problem, solution)
        min_step = 0
        distance = min_distance

        state = env_generate_state()
        env_start_time = datetime.datetime.now()
        episode = []
        current_best_distances = []
        start_distance = distance
        current_distances = []
        start_distances = []
        improving_steps = []
        improving_distances = []
        improving_percentage=[]

        inference_time = 0
        gpu_inference_time = 0
        env_act_time = 0
        no_improvement = 0
        loop_action = 0
        num_random_actions = 0
        alll=0
        nipv=0
        invalid_dim=0
        for action_index in range(len(action_timers)):
            action_timers[action_index] = 0.0
        for step in range(config.max_rollout_steps):
            #print(step)
            start_timer = datetime.datetime.now()
            if config.use_cyclic_rollout:
                choices = [1, 3, 4, 5, 8]
                if no_improvement == len(choices) + 1:
                    action = 0
                    no_improvement = 0
                else:
                    action = choices[loop_action]
                    loop_action += 1
                    if loop_action == len(choices):
                        loop_action = 0
            elif config.use_random_rollout:
                action = random.randint(0, config.num_actions - 1)
            else:
                gpu_start_time = datetime.datetime.now()
                action_pred,path_probs = policy_estimator.predict([state], [embedded_trip], embedded_paths_all,last_path_aim_1,last_path_aim_2,sess)
                #print(action_pred)
                #print(path_probs)
                action_pred = action_pred[0] + 1
                path_probs = path_probs[0]
                gpu_inference_time += (datetime.datetime.now() - gpu_start_time).total_seconds()
                '''
                action_probs = action_probs[0]
                path_probs = path_probs[0]
                history_action_probs = np.zeros(len(action_probs))
                action_prob_sum = 0.0
                for action_prob_index in range(len(action_probs)):
                    action_prob_sum += action_probs[action_prob_index]
                for action_prob_index in range(len(action_probs)):
                    action_probs[action_prob_index] /= action_prob_sum
                '''
                path_prob_sum = 0.0
                for path_prob_index in range(len(path_probs)):
                    path_prob_sum += path_probs[path_prob_index]
                for path_prob_index in range(len(path_probs)):
                    path_probs[path_prob_index] /= path_prob_sum
                
                if config.use_history_action_distribution and (index_sample > 0):
                    history_action_count_sum = 0
                    for action_count_index in range(len(action_probs)):
                        history_action_count_sum += count_timers[action_count_index + 1]
                    for action_count_index in range(len(action_probs)):
                        history_action_probs[action_count_index] = count_timers[action_count_index + 1]/history_action_count_sum
                        action_probs[action_count_index] = action_probs[action_count_index]/2 + history_action_probs[action_count_index]/2


                if config.use_rl_loss:
                    states.append(state)
                    trips.append(embedded_trip)
                elif random.uniform(0, 1) < 0.05:
                    action_label = [0] * config.num_actions
                    action_index = 0
                    min_action_time = sys.maxint
                    rewards = []
                    action_times = []
                    for action_to_label in range(1, config.num_actions):
                        action_start_time = datetime.datetime.now()
                        _, reward, _, _, _ = env_step(step, state, problem, min_distance, solution, distance, action_to_label, False)
                        action_time = (datetime.datetime.now() - action_start_time).total_seconds()
                        rewards.append(reward)
                        action_times.append(action_time)
                        if reward > EPSILON and action_time < min_action_time:
                            action_index = action_to_label
                            min_action_time = action_time
                            break
                    action_label[action_index] = 1
                    states.append(state)
                    trips.append(embedded_trip)
                    action_labels.append(action_label)

                if (config.model_to_restore is not None and should_restart(min_distance, distance, no_improvement)) or no_improvement >= config.max_no_improvement:
                    action = 0
                    no_improvement = 0
                else:
                    #if np.random.uniform() < config.epsilon_greedy*(1.2*config.num_episode-index_sample)/(1.2*config.num_episode):
                    if np.random.uniform() < config.epsilon_greedy:
                        action = np.random.randint(config.num_actions - 1) + 1
                        num_random_actions += 1
                    else:
                        action = int(action_pred)
                        '''
                        if config.sample_actions_in_rollout:
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) + 1
                            #path_aims = np.random.choice(np.arange(len(path_probs)),2, replace=False,p=path_probs)
                            #path_aim_1=path_aims[0]
                            #path_aim_2=path_aims[1]
                        else:
                            action = np.argmax(action_probs) + 1
                        '''
                    path_aims = np.random.choice(np.arange(len(path_probs)),2, replace=False,p=path_probs)
                    path_aim_1_label=path_aims[0]
                    path_aim_2_label=path_aims[1]
                    path_prob_1=path_probs[path_aim_1_label]
                    path_prob_2=path_probs[path_aim_2_label]
                    if config.one_vehicle_mode and action not in [1,2,3,12,25,26,27]:
                        tmppr=np.random.uniform()
                        if tmppr<0.05:
                            action = 12
                        elif tmppr<0.10:
                            action = 26
                        elif tmppr<0.20:
                            action =25
                        elif tmppr<0.25:
                            action =12
                        else:
                            action = action%3+1
                    #print(action)
                    #print(step)
            end_timer = datetime.datetime.now()
            inference_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            #if index_sample>=1 and np.random.uniform()<0.6:
            #    solution=[[[1,2],[2,3]]]
            if  len(np.array(solution).shape)!=1:
                print("invalid dimension")
                #scio.savemat('invalid.mat',{'solution':solution,'step':index_sample})
                #print(solution)
                invalid_dim=1
                break
            #path_aim_1_embed = [embed_path_with_nodes(problem,solution[path_aim_1])]
            #path_aim_1_embed.append(embed_path_with_nodes(problem,solution[path_aim_2]))
            #print(np.array(path_aim_1_embed).shape,'thisone')
            #path_aim_1_attention = embedding_net(path_aim_1_embed)
            #print(path_aim_1_attention,'pata')

            delta_to_compare=[]
            if config.with_ranking:
                path_aim_2_tmp=path_aim_2_label
                for path_aim_1_tmp in range(config.num_paths):
                    if path_aim_1_tmp != path_aim_1_label:
                        if action not in [1, 2, 3, 4,5,6,11, 16] and path_aim_1_tmp == path_aim_2_tmp:
                            delta_to_compare.append([path_aim_1_tmp,0.0])
                        else:
                            _,delta_tmp = improve_solution_by_action(step, problem, solution, action,path_aim_1_tmp,path_aim_2_tmp,True)
                            delta_to_compare.append([path_aim_1_tmp,delta_tmp])

            next_state, reward, done, next_solution, next_distance = env_step(step, state, problem, min_distance, solution, distance, action,path_aim_1_label,path_aim_2_label)
            
            if config.with_ranking:
                rank = 1
                prob_rank=[0]*config.num_paths
                for delta_label in range(config.num_paths-1):
                    if delta_to_compare[delta_label][1]<0.0-reward-1e-10:
                        rank+=1
                #Ranks[rank-1]+=1
                delta_to_compare.append([path_aim_1_label,0.0-reward])
                delta_to_compare.sort(key=takeSecond)
                #print(delta_to_compare)
                if reward > EPSILON:
                    #print("show")
                    Ranks[rank-1]+=1
                    for delta_label in range(config.num_paths):
                        prob_rank[delta_label]=path_probs[delta_to_compare[delta_label][0]]
                    Prob_ranks.append(prob_rank)

            actions_record[action]+=1
            #if action==11 or action in range(20,25):
            #    alll+=1
            if next_distance >= distance - EPSILON:
                no_improvement += 1
                #if action==11 or action in range(20,25):
                #    nipv+=1
            else:
                #TODO
                no_improvement = 0

            current_distances.append(distance)
            start_distances.append(start_distance)
            if action == 0:
                start_distance = next_distance
            current_best_distances.append(min_distance)
            
            #print(action)
            #print(next_solution)
            #print(next_distance)
            #print(min_distance)
            if next_distance < min_distance - EPSILON and not is_illegal_solution(problem,next_solution):
                stepss.append(step)
                #print("YES")
                #print(stepss)
                min_distance = next_distance
                min_distances.append(min_distance)
                #print(min_distances)
                min_step = step
                best_solution = copy.deepcopy(next_solution)
            if (step + 1) % config.step_interval == 0:
                print('rollout_num={}, index_sample={}, min_distance={}, min_step={}'.format(
                    step + 1, index_sample, min_distance, min_step
                ))
                temp_timers = np.asarray(action_timers)
                temp_count_timers = temp_timers[::2]
                temp_time_timers = temp_timers[1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in temp_time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in temp_count_timers])))
            if done:
                #print("hh")
                break


            episode.append(Transition(
                state=state, trip=copy.deepcopy(embedded_trip), next_distance=next_distance,
                action=action, path_aim_1=path_prob_1, path_aim_2=path_prob_2, reward=reward, next_state=next_state, done=done))
            last_path_aim_1 = [embed_path_with_nodes(problem,solution[path_aim_1_label],embedded_trip)]
            last_path_aim_2 = [embed_path_with_nodes(problem,solution[path_aim_2_label],embedded_trip)]
            embedded_paths_all = embed_solution_with_path(problem,solution,embedded_trip)
            state = next_state
            solution = next_solution
            #solution = sort_solution(problem,solution)
            if config.use_attention_embedding:
                embedded_trip = embed_solution_with_attention(problem, solution)
            else:
                embedded_trip = [0]
            distance = next_distance
            end_timer = datetime.datetime.now()
            env_act_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            if reward > 0 and action != 0:
                #print(len(improving_steps))
                #print("in")
                improving_steps.append(step)
                improving_distances.append(distance)
                if step!=0:
                    improving_percentage.append(len(improving_steps)/step)
            if len(improving_steps)>=1.1*config.batch_size :
                #scio.savemat('a.mat',{'improving_steps':improving_steps,'distances':improving_distances,'percs':improving_percentage})
                #print(improving_steps)
                #print(improving_distances)
                #print(step)
                #main_actions.append(actions_record)
                #main_percentages.append(improving_percentage[-1])
                #print(alll,'all')
                #print(nipv,'nipv')
                #scio.savemat('b_perturb_10_nos_act_line2.mat',{'main_steps':main_steps,'main_distances':main_distances,'percs':main_percentages,'actions':main_actions})
                break
            elif min_distance<=test_distances[index_sample%5]:
                break
            

        if(invalid_dim==1):
            continue
        if(index_sample in [5,6,7]) and config.with_test_set:
            test_distances[index_sample%5]=min_distance
            print(test_distances)
        if config.use_random_rollout:
            temp = np.inf
            for rollout_step in range(num_checkpoint):
                current_region_min_step = np.argmin(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)]) + rollout_step * config.step_interval
                current_region_min_distance = min(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)])
                if temp > current_region_min_distance:
                    distance_record[1, rollout_step] = current_region_min_distance
                    step_record[1, rollout_step] = current_region_min_step
                    temp = current_region_min_distance
                else:
                    distance_record[1, rollout_step] = distance_record[1, rollout_step - 1]
                    step_record[1, rollout_step] = step_record[1, rollout_step - 1]

        start_timer = datetime.datetime.now()
        distances.append(min_distance)
        steps.append(min_step)
        if validate_solution(problem, best_solution, min_distance) and not config.with_test_set:
            main_actions.append(actions_record)
            main_percentages.append(improving_percentage[-1])
            main_steps.append(stepss)
            main_distances.append(calculate_solution_distance(problem, best_solution))
            #scio.savemat('b_attention_pred_with_ranking_line3.mat',{'main_steps':main_steps,'main_distances':main_distances,'percs':main_percentages,'actions':main_actions})
            if config.with_ranking:
                Ranks_main.append(Ranks)
                Prob_ranks = np.average(Prob_ranks,axis=0)
                Prob_ranks_main.append(Prob_ranks)
                #scio.savemat('d_attention_ranking_line3.mat',{'Ranks':Ranks_main,'Prob_ranks':Prob_ranks_main})
            solutions.append(best_solution)
            print('solution={}'.format(best_solution))
            #for path in best_solution:
            #    print(calculate_path_distance(problem,path))
            #print(calculate_solution_distance(problem,best_solution))
        else:
            print(best_solution)
        #    print('invalid solution')
        if validate_solution(problem,best_solution,min_distance) and index_sample%8>=5 and config.with_test_set:
            main_actions_problem.append(actions_record)
            main_steps_problem.append(stepss)
            main_distances_problem.append(calculate_solution_distance(problem, best_solution))
            #solutions.append(best_solution)
            print('solution={}'.format(best_solution))

        if not (config.use_cyclic_rollout or config.use_random_rollout):
            future_best_distances = [0.0] * len(episode)
            future_best_distances[-1] = episode[len(episode) - 1].next_distance
            step = len(episode) - 2
            while step >= 0:
                if episode[step].action != 0:
                    future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
                else:
                    future_best_distances[step] = current_distances[step]
                step = step - 1

            historical_baseline = None
            for t, transition in enumerate(episode):
                # total_return = sum(config.discount_factor**i * future_transition.reward for i, future_transition in enumerate(episode[t:]))
                if historical_baseline is None:
                    if transition.action == 0:
                        #TODO: dynamic updating of historical baseline, and state definition
                        historical_baseline = -current_best_distances[t]
                        # historical_baseline = 1/(current_best_distances[t] - 10)
                    actions.append(0)
                    advantages.append(0)
                    path_aims_1.append(0.0)
                    path_aims_2.append(0.0)
                    continue
                # if transition.action == 0:
                #     historical_baseline = -current_distances[t]
                if transition.action > 0:
                    # total_return = transition.reward
                    if transition.reward < EPSILON:
                        total_return = -1.0
                    else:
                        total_return = 1.0
                    #     total_return = min(transition.reward, 2.0)
                    # total_return = start_distances[t] - future_best_distances[t]
                    # total_return = min(total_return, 1.0)
                    # total_return = max(total_return, -1.0)
                    total_return = -future_best_distances[t]
                    # total_return = 1/(future_best_distances[t] - 10)
                else:
                    if transition.state[-1] != 0 and transition.state[-2] < 1e-6:
                        # if future_best_distances[t] < current_best_distances[t] - 1e-6:
                        total_return = 1.0
                    else:
                        total_return = -1.0
                    total_return = 0
                    actions.append(0)
                    advantages.append(0)
                    path_aims_1.append(0.0)
                    path_aims_2.append(0.0)
                    continue
                # baseline_value = value_estimator.predict(states)
                # baseline_value = 0.0
                baseline_value = historical_baseline
                advantage = total_return - baseline_value
                actions.append(transition.action)
                path_aims_1.append(transition.path_aim_1)
                path_aims_2.append(transition.path_aim_2)
                advantages.append(advantage)
                # value_estimator.update(states, [[total_return]], sess)

            states = np.reshape(np.asarray(states), (-1, env_observation_space_n)).astype("float32")
            if config.use_attention_embedding:
                trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, config.input_embedded_trip_dim_2)).astype("float32")
            actions = np.reshape(np.asarray(actions), (-1))
            advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")
            path_aims_1 = np.reshape(np.asarray(path_aims_1), (-1))
            path_aims_2 = np.reshape(np.asarray(path_aims_2), (-1))
            #R_anals_actions.append(actions)
            #R_anals_path_aims_1.append(path_aims_1)
            #R_anals_path_aims_2.append(path_aims_2)
            #scio.savemat('R_anals.mat',{'R_actions':R_anals_actions,'R_path1':R_anals_path_aims_1,'R_path2':R_anals_path_aims_2})
            if config.use_rl_loss:
                print('num_random_actions={}'.format(num_random_actions))
                print('actions={}'.format(actions[:100]).replace('\n', ''))
                print('aims_1={}'.format(path_aims_1[:100]).replace('\n', ''))
                print('aims_2={}'.format(path_aims_2[:100]).replace('\n', ''))
                print('advantages={}'.format(advantages[:100]).replace('\n', ''))
                if config.model_to_restore is None and index_sample <= config.max_num_training_epsisodes:
                    filtered_states = []
                    filtered_trips = []
                    filtered_advantages = []
                    filtered_actions = []
                    filtered_path_aims_1 = []
                    filtered_path_aims_2 = []
                    end = 0
                    for action_index in range(len(actions)):
                        if actions[action_index] > 0:
                            filtered_states.append(states[action_index])
                            filtered_trips.append(trips[action_index])
                            filtered_advantages.append(advantages[action_index])
                            filtered_actions.append(actions[action_index] - 1)
                            filtered_path_aims_1.append(path_aims_1[action_index])  
                            filtered_path_aims_2.append(path_aims_2[action_index])

                        else:
                            num_bad_steps = config.max_no_improvement
                            end = max(end, len(filtered_states) - num_bad_steps)
                            filtered_states = filtered_states[:end]
                            filtered_trips = filtered_trips[:end]
                            filtered_advantages = filtered_advantages[:end]
                            filtered_actions = filtered_actions[:end]
                            filtered_path_aims_1 = filtered_path_aims_1[:end]
                            filtered_path_aims_2 = filtered_path_aims_2[:end]

                    filtered_states = filtered_states[:end]
                    filtered_trips = filtered_trips[:end]
                    filtered_advantages = filtered_advantages[:end]
                    filtered_actions = filtered_actions[:end]
                    filtered_path_aims_1 = filtered_path_aims_1[:end]
                    filtered_path_aims_2 = filtered_path_aims_2[:end]

                    num_states = len(filtered_states)
                    if config.use_attention_embedding and num_states > config.batch_size:
                        downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                        #print(sorted(downsampled_indices))
                        filtered_states = np.asarray(filtered_states)[downsampled_indices]
                        filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                        filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                        filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
                        filtered_path_aims_1 = np.asarray(filtered_path_aims_1)[downsampled_indices]
                        filtered_path_aims_2 = np.asarray(filtered_path_aims_2)[downsampled_indices]

                    if(end):
                        loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions,filtered_path_aims_1,filtered_path_aims_2, sess)
                        print('loss={}'.format(loss))
                    else:
                        print("no update this episode")
                else:
                    print("no update just test")
                    if(index_sample%8==7):
                        scio.savemat('c{}_test.mat'.format(index_sample%8),{'main_steps':main_steps_problem,'main_actions':main_actions_problem,'main_distances':main_distances_problem})

            else:
                #TODO: filter and reshape
                action_labels = np.reshape(np.asarray(action_labels), (-1, config.num_actions))
                loss, accuracy = policy_estimator.train(states, trips, action_labels, sess)
                print('loss={}, accuracy={}'.format(loss, accuracy))
        timers_epoch = [inference_time, gpu_inference_time, env_act_time, (datetime.datetime.now() - start_timer).total_seconds()]
        timers_epoch.extend(action_timers)
        timers.append(timers_epoch)
        if config.model_to_restore is None and index_sample > 0 and index_sample % 1498 == 0:
            save_path = saver.save(sess, "./rollout_model_{}_{}_{}_1500_epoch_attention_with_score.ckpt".format(index_sample, config.num_history_action_use, config.max_rollout_steps))
            print("Model saved in path: %s" % save_path)
    # save_path = saver.save(sess, "./rollout_model.ckpt")
    # print("Model saved in path: %s" % save_path)

    main_steps_max=[]
    for stepss in main_steps:
        #print(stepss)
        main_steps_max.append(stepss[-1])
    #scio.savemat('b_perturb_10_nos_act.mat',{'main_steps':main_steps_max,'main_distances':main_distances,'percs':main_percentages,'actions':main_actions})
    #print(main_distances)
    #print(main_percentages)
    #scio.savemat('b_attention_pred.mat',{'main_steps':main_steps,'main_distances':main_distances,'percs':main_percentages,'actions':main_actions})
    #scio.savemat('R_anals.mat',{'R_actions':R_anals_actions,'R_path1':R_anals_path_aims_1,'R_path2':R_anals_path_aims_2})
    solution_after=solutions[0]
    #save_solution_bef(problem, solution_before)
    #save_solution_aft(problem, solution_after)
    print(solutions,'sols')
    print(calculate_solution_distance_dep(problem,solution_after))
    calculate_outside_ratio(problem,solution_after)
    print('solving time = {}'.format(datetime.datetime.now() - start))
