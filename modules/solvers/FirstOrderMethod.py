# First Order Method template, to be used with a particular optimizer

import numpy as np
import time
import logging

class FirstOrderMethod():
    def __init__(self, game, LocalOpt):
        self.game = game
        self.name = 'fom'
        self.LocalOpt = LocalOpt

        if not hasattr(self.game, 'infosets_dict'):
            print("Getting game tree")
            self.game.get_game_tree()
            print("Done getting game tree")

    def add_infoset(self, info_strat):
        self.infosets_strat.append(info_strat)
        self.infoset_util_vecs.append(np.zeros_like(info_strat.x))

    def make_utils(self, node, reach=1):
        ev = 0
        if node.type == "leaf":
            ev = node.payoffs[0]
        elif node.type == "chance":
            for ai, action in enumerate(node.actions):
                child = node.gotochild(action)
                prob = node.action_probs[action]
                child_ev = self.make_utils(child, reach * prob)
                ev += prob * child_ev
        else:
            assert node.type == "player"
            inf_id = node.infoset.id
            for ai, action in enumerate(node.actions):
                child = node.gotochild(action)
                prob = self.infosets_strat[inf_id].x[ai]
                child_ev = self.make_utils(child, reach * prob)
                ev += prob * child_ev
                #you are taking ai with probability 1, so no need to multiply it by prob:
                self.infoset_util_vecs[inf_id][ai] += child_ev * reach
        return ev

    def solve(self, init = "uniform", max_iter=1000000000, timelimit=10800, tol=0, reporting={'report_style': "list", 'report_freq': 50}):
        
        history = None
        if isinstance(reporting, dict):
            report_style = reporting['report_style']
            freq = reporting['report_freq']   
            if report_style == "list":
                history = {"iter": [], "gap": []}
        else:
            report_style = None
            freq = 2**10
            
        
        self.infosets_strat = []
        self.infoset_util_vecs = []

        if isinstance(init, np.ndarray):
            i = 0
            for key in self.game.infoset_keys:
                num_acts = self.game.infosets_dict[key].num_actions
                self.add_infoset(self.LocalOpt(num_acts, init=init[i:i+num_acts]))
                i += num_acts
            assert i == len(init)
        else:
            for key in self.game.infoset_keys:
                self.add_infoset(self.LocalOpt(self.game.infosets_dict[key].num_actions, init=init))

        root = self.game.nodes_dict["/"]

        # print("Starting solver")
        start_time = time.time()
        for iter in range(max_iter):
            gap = 0
            root_ev = self.make_utils(root)
            for (info_strat, action_utils) in zip(self.infosets_strat, self.infoset_util_vecs):
                # print(action_utils)
                br = action_utils.max()
                inf_ev = info_strat.step(action_utils)
                gap = max(gap, br - inf_ev)
                action_utils *= 0
            # print("t", iter, "ev", root_ev, "gap", gap)
            if report_style and iter % freq == 0:
                if report_style == "logging":
                    logging.info(f"iter {iter} | time {time.time()-start_time} | value {root_ev} | gap {gap}")
                elif report_style == "list":
                    history["iter"].append(iter)
                    history["gap"].append(gap)
                    print(iter, " : ", gap)
                else:
                    raise ValueError("report_style not recognized")
            
            #gap is L_infinity norm of projected gradient / KKT violation of sorts
            if gap <= tol: break
            if time.time() - start_time >= timelimit: break
        
        end_time = time.time()

        if report_style == "logging":
            logging.info(f"iter {iter} | time {end_time-start_time} | value {root_ev} | gap {gap}")
        elif report_style == "list":
            history["iter"].append(iter)
            history["gap"].append(gap)
            print(iter, " : ", gap)
        
        x = []
        for info_strat in self.infosets_strat:
            x.extend(info_strat.last_x)

        return {
            "x": x,
            "value": root_ev,
            "iter": iter,
            "time": end_time - start_time,
            "gap": gap,
            "hist": history
        }
