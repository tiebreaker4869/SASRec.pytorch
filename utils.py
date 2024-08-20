import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import List, Tuple, Dict
import json

def build_pop_list(dataset_name: str) -> defaultdict:
    """
    Build item popularity list.
    Args:
        dataset_name: dataset name, assume the user id and item id start from 1
    Returns:
        pop_list: item popularity list, key is item id and value is the number of interactions.
    """
    ui_mat = np.loadtxt("data/%s.txt" % dataset_name, dtype=np.int32)
    pop_list = defaultdict(int)
    for ui_pair in ui_mat:
        pop_list[ui_pair[1]] += 1
    return pop_list


def build_index(
    dataset_name: str, meta_info: Dict[str, int] = None
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build user to item and item to user relationship.
    Args:
        dataset_name: dataset name, assume the user id and item id start from 1
        meta_info: a dict containing meta info such as num_users and num_items. Useful when finetuning from a checkpoint.
    Returns:
        u2i_index: user item relationship, specify the items each user interacted with.
        i2u_index: item user relationship, specify the users each item was interacted by.
    """
    ui_mat = np.loadtxt("data/%s.txt" % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    if meta_info:
        n_users = max(n_users, meta_info["num_users"])
        n_items = max(n_items, meta_info["num_items"])

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index


# sampler for batch generation
def random_neq(l: int, r: int, s) -> int:
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(
    user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED
):
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
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        User,
                        usernum,
                        itemnum,
                        batch_size,
                        maxlen,
                        self.result_queue,
                        np.random.randint(2e9),
                    ),
                )
            )
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
    f = open("data/%s.txt" % fname, "r")
    for line in f:
        u, i = line.rstrip().split(" ")
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 13:  # 10 for testset, 1 for valid set, 1 for training set
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-11]
            user_valid[user] = []
            user_valid[user].append(User[user][-11])
            user_test[user] = []
            user_test[user].extend(User[user][-10:])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate_serendipity(model, dataset, args, pop_list):
    
    top_100_pop = sorted(pop_list.items(), key=lambda x: x[1], reverse=True)[:100]
    
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    hit = 0.0
    ndcg = 0.0
    
    valid_user = 0.0
    
    users = range(1, usernum + 1)
    
    for user in users:
        if user not in train or len(train[user]) < 1 or user not in test or len(test[user]) < 1:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[user][0]
        idx -= 1
        
        for i in reversed(train[user]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        rated = set(train[user])
        rated.add(0)
        
        # find the diff of test set and top 100 popular items
        test_set = set(test[user])
        top_100_pop_set = set([item[0] for item in top_100_pop])
        diff = test_set - top_100_pop_set
        if len(diff) == 0:
            continue
        
        item_idx = [random.choice(list(diff))]
        
        for _ in range(1000):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        predictions, _ = model.predict(
            *[np.array(l) for l in [[user], [seq], item_idx]], topk=args.save_topk
        )
        
        predictions = -predictions
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()
        
        valid_user += 1
        
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            hit += 1
        
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()
        
    return ndcg / valid_user, hit / valid_user


# TODO: merge evaluate functions for test and val set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)
    
    # sample 10000 users for evaluation
    users = random.sample(users, 10000)
    
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
        # item_idx = [test[u][0]]
        item_idx = [random.choice(test[u])]
        for _ in range(1000):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # Model prediction
        predictions, _ = model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]], topk=args.save_topk
        )
        predictions = -predictions
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if u not in train or len(train[u]) < 1 or u not in valid or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(1000):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions, _ = model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]], topk=-1
        )
        predictions = -predictions
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def infer_and_save(model, dataset, args):
    print("Inferencing and saving results...")

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    users = range(1, usernum + 1)

    all_result = []

    for u in users:
        if (
            u not in train
            or len(train[u]) < 1
            or u not in test
            or len(test[u]) < 1
            or u not in valid
            or len(valid[u]) < 1
        ):
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        item_idx = list(range(1, itemnum + 1))

        _, topk_indices = model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]], topk=args.save_topk
        )

        # save topk items
        topk_items = [item_idx[i] for i in topk_indices[0]]

        # save histories, topk items, and ground truth
        all_result.append((train[u] + valid[u], topk_items, test[u]))

    # save all results
    with open("result.json", "w") as f:
        json.dump(all_result, f)

    print("Results saved to result.json")
