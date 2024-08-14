import sys
import copy
import torch
import random
import json
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import List, Tuple, Dict

def build_index(dataset_name: str, meta_info: Dict[str, int]=None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build user to item and item to user relationship.
    Args:
        dataset_name: dataset name, assume the user id and item id start from 1
        meta_info: a dict containing meta info such as num_users and num_items. Useful when finetuning from a checkpoint.
    Returns:
        u2i_index: user item relationship, specify the items each user interacted with.
        i2u_index: item user relationship, specify the users each item was interacted by.
    """
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    
    if meta_info:
        n_users = max(n_users, meta_info['num_users'])
        n_items = max(n_items, meta_info['num_items'])

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        # while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)
        while uid not in user_train or len(user_train[uid]) < 1:
            uid = random.choice(list(user_train.keys()))

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#     ### 初始化 topk dict ###
#     topk_dict = {}
#     ### 初始化 topk dict ###

#     #if usernum>10000:
#     #    users = random.sample(range(1, usernum + 1), 10000)
#     #else:
#     #    users = range(1, usernum + 1)
#     users = range(1, usernum + 1)
#     for u in users:

#         if u not in train or len(train[u]) < 1 or u not in test or len(test[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions, topk_indices = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], topk=args.save_topk)
#         predictions = -predictions
#         predictions = predictions[0] # - for 1st argsort DESC

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1

#         ### 保存 top-k 推荐结果 ###
#         topk_items = [item_idx[i] for i in topk_indices[0].cpu().numpy()]
#         interaction_history = [i for i in train[u]]
#         topk_dict[u] = (interaction_history, topk_items)
#         ### 保存 top-k 推荐结果 ###
        
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()
#     ### 保存 top-k 推荐结果到 CSV 文件 ###
#     with open('topk_recommendations.csv', 'w', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(['user_id', 'interaction_history', 'recommendations'])
#         for user, (history, recommendations) in topk_dict.items():
#             csvwriter.writerow([user, ' '.join(map(str, history)), ' '.join(map(str, recommendations))])
#     ### 保存 top-k 推荐结果到 CSV 文件 ###

#     return NDCG / valid_user, HT / valid_user

# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0
#     topk_dict = {}

#     users = range(1, usernum + 1)
#     for u in users:
#         if u not in train or len(train[u]) < 1 or u not in test or len(test[u]) < 1:
#             continue

#         # Prepare sequence for prediction
#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1:
#                 break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated:
#                 t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         # Model prediction
#         predictions, topk_indices = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], topk=args.save_topk)
#         predictions = -predictions
#         predictions = predictions[0]
#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1

#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user

def evaluate(model, dataset, args, save_path='evaluation_results.json'):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    topk_dict = {}

    # 用于保存每个用户的历史记录和推荐结果
    results = {}

    users = range(1, usernum + 1)
    for u in users:
        if u not in train or len(train[u]) < 1 or u not in test or len(test[u]) < 1:
            continue

        # Prepare sequence for prediction
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # Model prediction
        predictions, topk_indices = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], topk=args.save_topk)
        predictions = -predictions
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        # 保存每个用户的历史记录和推荐结果
        results[u] = {
            'sequence': seq.tolist(),
            'topk_recommendations': [item_idx[idx] for idx in topk_indices[0].tolist()],
            'true_item': test[u][0]
        }

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # 将结果保存到文件中
    with open(save_path, 'w') as f:
        json.dump(results, f)

    return NDCG / valid_user, HT / valid_user



# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if u not in train or len(train[u]) < 1 or u not in valid or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions, _ = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], topk=-1)
        predictions = -predictions
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
