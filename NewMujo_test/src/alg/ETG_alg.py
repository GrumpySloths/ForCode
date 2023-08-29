import numpy as np
import copy


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class SimpleGA:
    '''Simple Genetic Algorithm.'''

    def __init__(
        self,
        num_params,  # number of model parameters
        sigma_init=0.1,  # initial standard deviation
        sigma_decay=0.999,  # anneal standard deviation
        sigma_limit=0.01,  # stop annealing if less than this
        popsize=256,  # population size
        elite_ratio=0.1,  # percentage of the elites
        forget_best=False,  # forget the historical best elites
        weight_decay=0.01,  # weight decay coefficient
        param=None,
    ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        if param is None:
            self.best_param = np.zeros(self.num_params)
        else:
            self.best_param = param
        self.curr_best_param = self.best_param
        self.curr_best_reward = 0
        self.best_reward = 0

        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def reset(self, param):
        self.best_param = copy(param)
        self.curr_best_param = copy(param)
        self.first_iteration = True

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(self.popsize,
                                       self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            '''
            遗传算法中的杂交，让任意两个成员杂交生成新的成员
            '''
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            if self.first_iteration:
                solutions.append(self.best_param + self.epsilon[i])
            else:
                child_params = mate(self.elite_params[idx_a],
                                    self.elite_params[idx_b])
                solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize
                ), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            #将之前保存的elite和新的population一起进行比较，保证结果至少不会变差
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]
        self.curr_best_param = np.copy(self.elite_params[0])

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def get_best_param(self):
        return self.best_param

    def result(
        self
    ):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_param, self.best_reward, self.curr_best_reward,
                self.sigma, self.curr_best_param)