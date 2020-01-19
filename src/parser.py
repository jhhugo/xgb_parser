# -*- coding: utf-8 -*-
from pathlib import Path
import json
from collections import deque, defaultdict
from joblib import Parallel, delayed
import gc
from itertools import chain

class TreeParser(object):
    def __init__(self, path, njobs=1):
        if not isinstance(path, str):
            raise TypeError("please input str path")
        self.path = Path(path)
        self.njobs = njobs
        # read json
        with self.path.open('r', encoding='utf-8') as f:
            self.tree = json.load(f)
        self.data = self._get_data(self.tree)

    def _sub_parser(self, start, subtrees):
        res = []
        for subtree in subtrees:
            # put head
            queue = deque([subtree])
            rule = defaultdict(list)
            leaves = 0
            while queue:
                node = queue.popleft()
                if 'children' in node.keys():
                    cond = node['split']
                    rule[cond].append(node['split_condition'])
                    childs = node['children']
                    queue.extend(childs)
                elif 'leaf' in node.keys():
                    leaves += 1
                else:
                    raise ValueError("Invalid tree json, please check again!")

            # sorted
            rule = {k: sorted(v) for k, v in rule.items()}
            # tree id, rule, leaves    
            res.append((start, rule, leaves))
            del rule, queue, leaves
            gc.collect()
            start += 1
        return res
    
    def _get_data(self, tree):
        m = len(tree)
        step =  m // self.njobs
        trees = [(start, tree[start: start + step]) for start in range(0, m, step)]
        res = Parallel(n_jobs=self.njobs,)(delayed(self._sub_parser)(start, subtrees,) for start, subtrees in trees)
        res = list(chain(*res))
        res = sorted(res, key=lambda s: s[0])
        return res
    
    def get_rule_leaf(self, topk=1):
        if topk < 0:
            topk = len(self.data) + topk + 1
        
        all_leaf = 0
        combine_rule = defaultdict(set)
        for idx, rule, leaves in self.data:
            all_leaf += leaves
            if idx < topk:
                for key, values in rule.items():
                    combine_rule[key] = combine_rule[key].union(values)
        return combine_rule, all_leaf
    