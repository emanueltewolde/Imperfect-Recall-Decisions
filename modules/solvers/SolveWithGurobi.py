# Gurobi solver

import numpy as np
import gurobipy as gp

class solve_with_gurobi():
    def __init__(self, game):
        self.game = game
        self.name = 'gurobi'


        if not hasattr(self.game, 'infosets_dict'):
            print("Getting game tree")
            self.game.get_game_tree()
            print("Done getting game tree")

    def make_objective(self, node, reach=1, chance_reach=1):
        if node.type == "leaf":
            if node.payoffs[0] * chance_reach != 0:
                self.objective.append((node.payoffs[0] * chance_reach, reach))
        elif node.type == "chance":
            for action in node.actions:
                self.make_objective(node.gotochild(action), reach, chance_reach * node.action_probs[action])
        else:
            assert node.type == "player"
            for ai, action in enumerate(node.actions):
                child = node.gotochild(action)
                child_reach = self.model.addVar(name=child.descr)
                self.model.addConstr(child_reach == reach * self.infoset_xs[node.infoset.descr][ai])
                self.make_objective(child, child_reach, chance_reach)


    # 86400 seconds = 24h
    def solve(self, timelimit=86400, tol=1e-6, output_settings=1):
        self.model = gp.Model()
        self.model.setParam("TimeLimit", timelimit)
        self.model.setParam("OptimalityTol", tol)
        self.model.setParam("LogToConsole", output_settings)

        # make variables
        self.infoset_xs = {}

        for key in self.game.infoset_keys:
            infoset = self.game.infosets_dict[key]
            infoset_x = [self.model.addVar(name=key+"/"+action) for action in infoset.actions]
            self.infoset_xs[infoset.descr] = infoset_x
            self.model.addConstr(np.ones(infoset.num_actions) @ infoset_x == 1)

        # walk the tree
        self.objective = []
        self.make_objective(self.game.nodes_dict["/"])
        self.model.setObjective(gp.LinExpr(self.objective), gp.GRB.MAXIMIZE)
        self.model.setParam("NonConvex", 2)
        self.model.optimize()

        x = []
        for key in self.game.infoset_keys:
            infoset = self.game.infosets_dict[key]
            x.extend(v.x for v in self.infoset_xs[infoset.descr])

        return {
            "x": x,
            "time": self.model.Runtime,
            "value": self.model.getAttr("ObjVal"),
            "status": self.model.getAttr("Status"),
        }
