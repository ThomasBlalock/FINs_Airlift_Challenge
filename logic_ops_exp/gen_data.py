import random as rand
import numpy as np
PROP = 3
F = 0
T = 1
IDK = 2

IMP = 0
AND = 1
OR = 2

def gen_data(i: int, key_size: int):
    # [prop?, T/F, T/F, ID, op, T/F, ID]

    op_list = [IMP, AND, OR]
    num_ops = len(op_list)

    # gen random operations
    exps = []
    exp_list = [
        (implication(), 2),
        (contrapositive(), 2),
    ]
    num_exps = len(exp_list)
    prop = 0
    for _ in range(i):
        rand_op = rand.randint(0, num_exps-1)
        rand_truth_a = rand.randint(0, 1)
        rand_truth_b = rand.randint(0, 1)
        exps += exp_list[rand_op][0](rand_truth_a, rand_truth_b, prop)
        prop += exp_list[rand_op][1]

    # gen random keys
    prop_keys = []
    for i in range(prop):
        prop_keys.append(np.random.rand(key_size))
        prop_keys[i] /= np.linalg.norm(prop_keys[i])
        prop_keys[i] = prop_keys[i].tolist()
    op_keys = []
    for i in range(num_ops):
        op_keys.append(np.random.rand(key_size))
        op_keys[i] /= np.linalg.norm(op_keys[i])
        op_keys[i] = op_keys[i].tolist()

    # make input matrices
    prop_mtx = []
    for exp in exps:
        if exp[0]==PROP:
            row = [
                exp[1], # exp truth
                exp[2] # A truth
            ]
            row += prop_keys[exp[3]] # A
            row += op_keys[exp[4]] # prop
            row += [exp[5]] # B truth
            row += prop_keys[exp[6]] # B
            prop_mtx.append(row)
    op_mtx = []
    for op in op_list:
        row = []
        row += op_keys[op] # op key
        row += [1 if j==op else 0 for j in range(num_ops)] # 1-hot encoded
        op_mtx.append(row)
    goal_mtx = []
    labels_mtx = []
    for exp in exps:
        if exp[0]!=PROP:
            row = [
                exp[1], # exp truth
                exp[2] # A truth
            ]
            row += prop_keys[exp[3]] # A
            row += op_keys[exp[4]] # prop
            row += [exp[5]] # B truth
            row += prop_keys[exp[6]] # B
            goal_mtx.append(row)
            labels_mtx.append([1 if i==exp[0] else 0 for i in range(3)])

    return {
        'prop_mtx': prop_mtx,
        'op_mtx': op_mtx,
        'goal_mtx': goal_mtx,
        'labels_mtx': labels_mtx
    }
    

class implication:
    def __init__(self):
        pass
    def __call__(self, truth_a, truth_b, prop):
        # A->B, A : B
        A = prop
        B = prop + 1
        out = []
        out.append([PROP, T, truth_a, A, IMP, truth_b, B])
        out.append([PROP, T, truth_a, A, OR, truth_a, A])
        if rand.randint(1,2)>1.5:
            out.append([T, T, truth_b, A, AND, truth_b, B])
        else:
            out.append([F, T, abs(truth_b-1), A, AND, abs(truth_b-1), B])
        return out

class contrapositive:
    def __init__(self):
        pass
    def __call__(self, truth_a, truth_b, prop):
        # A->B, ~B : ~A
        A = prop
        B = prop + 1
        out = []
        out.append([PROP, T, truth_a, A, IMP, truth_b, B])
        out.append([PROP, T, abs(truth_b-1), B, AND, abs(truth_b-1), B])
        if rand.randint(1,2)>1.5:
            out.append([F, T, truth_a, A, AND, truth_a, A])
        else:
            out.append([T, T, abs(truth_a-1), A, AND, abs(truth_a-1), A])
        return out
