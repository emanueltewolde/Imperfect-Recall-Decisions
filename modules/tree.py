import numpy as np

class Node():
# Data structure of a node in the decision tree
    def __init__(self, description, parent):
        #description: string name of the node
        #parent: parent node. Enter 'root' if root node
        #type: whether it is terminal, or which player or chance is assigned to it
        self.descr = description
        self.parent = parent
        if self.parent == 'root':
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        self.children = {}

    
    def update_type(self,args={}):
        self.type = args['type']
        if self.type == 'player':
            self.player = args['player']    #integer
            self.actions = args['actions'] 
            self.infoset = args.get('infoset')  #returns None if infoset key is not in args
        elif self.type == 'chance':
            self.actions = args['actions']
            if isinstance(args['action_probs'], dict):
                self.action_probs = args['action_probs']
            else:
                self.action_probs = {}
                for act, prob in zip(self.actions, args['action_probs']):
                    self.action_probs[act] = prob   #supposed to be a rational
        elif self.type == 'leaf':
            self.payoffs = np.asarray(args['payoffs'])
        else:
            raise Exception("args['type'] is invalid", args['type'])

    def add_id(self, id):
        self.id = id

    def add_children(self):
        if not hasattr(self, 'type'):
            raise Exception("First you have to update the type of this node!")
        for act in self.actions:
            self.add_child(act)

    def add_child(self, action_descr):
        if action_descr in self.children:
             raise Exception("Child already exists!", action_descr)
        
        child_descr = self.get_child_descr(action_descr)
        
        self.children[action_descr] = Node(description=child_descr, parent=self)

        if action_descr not in self.actions:
            self.actions.append(action_descr)
    
    def get_child_descr(self,action_descr):
            if self.type == 'player':
                if self.descr == '/':
                    child_descr = '/P' + str(self.player) + ':' + action_descr
                else:
                    child_descr = self.descr + '/P' + str(self.player) + ':' + action_descr
            elif self.type == 'chance':
                if self.descr == '/':
                    child_descr = '/C' + ':' + action_descr
                else:
                    child_descr = self.descr + '/C' + ':' + action_descr
            else:
                raise ValueError("Either you have not updated the type of this node yet (output None), or this is a leaf and therefore should not have any children!", getattr(self, 'type', None))
            return child_descr

    def add_inf(self, infoset):
        self.infoset = infoset
    
    def gotochild(self, act):
        return self.children[act]
    
    def gotoparent(self):
        if self.parent == 'root':
            return self, 'root'
        else:
            return self.parent, 'was not root'

class InfoSet():
# Data structure of an infoset on the decision tree
    def __init__(self, descr, node_descrs):
        #description: string name of the infoset
        #node_descriptions: string names of the nodes of the infoset
        self.descr = descr
        self.node_descrs = node_descrs
        self.length = len(self.node_descrs)

    def add_id(self, id):
        self.id = id

    def add_pl_acts(self, player, actions):
        self.player = player
        self.actions = actions
        self.num_actions = len(self.actions)
    
    def add_vars(self,vars):
        self.variables = vars

    def add_node(self, node):
        if node.descr not in self.node_descrs:
            if node.player == self.player and node.actions == self.actions:
                self.node_descrs.append(node.descr)
                self.length += 1
            else:
                raise Exception("The node you are trying to add to the infoset does not fit to the player identity or action set of the infoset!", node.descr, self.descr, node.player, self.player, node.actions, self.actions)


def create_infoset(nodes, descr, id):
    infoset = InfoSet(descr, nodes.keys())
    infoset.add_id(id)

    actions = []
    player = None
    num_changes_actions = 0
    num_changes_player = 0

    for node_descr, current in nodes.items():
        if not hasattr(current, 'type'):
            raise Exception("First you have to update the type of this node before dealing with this infoset!", node_descr)
        
        current.add_inf(infoset)

        if actions != current.actions:
            actions = current.actions
            num_changes_actions += 1                

        if player != current.player:
            player = current.player
            num_changes_player += 1                

    if num_changes_actions != 1:
        raise Exception("The info set named as the following has an ill-defined action set: ", descr)
    if num_changes_player != 1:
        raise Exception("The info set named as the following has an ill-defined player assignment: ", descr)

    infoset.add_pl_acts(player, actions)

    return infoset