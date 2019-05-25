import random
from matplotlib.pyplot import pause
import networkx as nx
from classes import Buchi, Fsa, Rabin
import numpy as np
from scipy import misc
import visdom


class PlotDynamicAutomata(object):

    '''
    Uses visdom (https://github.com/facebookresearch/visdom) to dynamically update
    the automata graph visualization
    '''
    
    def __init__(self, automata, port=8097):
        self.automata = automata
        self.viz = visdom.Visdom(port=port)
        
        self.dot_g = nx.drawing.nx_pydot.to_pydot(self.automata.g)

        self.initialized = False
        self.win = None
        self.last_state = None
        self.last_edge = None
        # self.text_win = self.viz.text("starting")
        
    def update(self, current_state, src_and_dest=None):
        '''
        current_state: list of node names as a string
        src_and_dest: list of tuples (src_node, destination_node)
        '''
        if self.win is not None:
            self.last_state.obj_dict['attributes']['style'] = 'unfilled'
            self.last_edge.obj_dict['attributes']['color'] = 'black'

        current_node = self.dot_g.get_node(current_state)
        #current_node[0].add_style(style='filled')
        current_node[0].obj_dict['attributes']['style'] = 'filled'
        self.last_state = current_node[0]
                
        if src_and_dest:
            current_edge = self.dot_g.get_edge(src_and_dest)
            #current_edge[0].set_color('red')
            current_edge[0].obj_dict['attributes']['color'] = 'red'
            self.last_edge = current_edge[0]
                    
        dot_g_svg = self.dot_g.create(format='svg')
        # svgstr = dot_g_svg.decode('utf-8')
        # import re
        # svg = re.search('<svg .+</svg>', svgstr, re.DOTALL)
        if self.win is None:
            self.win = self.viz.svg(dot_g_svg.decode('utf-8'))
        else:
            self.viz.svg(dot_g_svg.decode('utf-8'), win=self.win)


if __name__ == "__main__":
    from lomap.classes import Fsa

    fsa = Fsa()
    fsa.from_formula("F a && F b")
    cls = PlotDynamicAutomata(fsa)
    while True:
        pass