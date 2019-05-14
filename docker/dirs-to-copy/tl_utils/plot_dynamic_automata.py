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
    
    def __init__(self, automata):
        self.automata = automata
        self.viz = visdom.Visdom()
        self.dot_g = []
        for aut in automata:
            self.dot_g.append(nx.drawing.nx_pydot.to_pydot(aut.g))

        self.init_plot = True
        self.win = []
        self.last_state = []
        self.last_edge = []
        self.text_win = self.viz.text("starting")
        
        self.nb_of_automata = len(automata)
        
    def update(self, current_state, src_and_dest=None):
        '''
        current_state: list of node names as a string
        src_and_dest: list of tuples (src_node, destination_node)
        '''
        
        if not self.init_plot:
            for i in range(self.nb_of_automata):
                self.last_state[i].obj_dict['attributes']['style'] = 'unfilled'
                self.last_edge[i].obj_dict['attributes']['color'] = 'black'

        for i in range(self.nb_of_automata):
            current_node = self.dot_g[i].get_node(current_state[i])
            #current_node[0].add_style(style='filled')
            current_node[0].obj_dict['attributes']['style'] = 'filled'
            if len(self.last_edge) == self.nb_of_automata:
                self.last_state[i] = current_node[0]
            else:
                self.last_state.append(current_node[0])
                
            if src_and_dest[i]:
                current_edge = self.dot_g[i].get_edge(src_and_dest[i])
                #current_edge[0].set_color('red')
                current_edge[0].obj_dict['attributes']['color'] = 'red'
                if len(self.last_edge) == self.nb_of_automata:
                    self.last_edge[i] = current_edge[0]
                else:
                    self.last_edge.append(current_edge[0])
                    
            dot_g_svg = self.dot_g[i].create(format='svg')
            # svgstr = dot_g_svg.decode('utf-8')
            # import re
            # svg = re.search('<svg .+</svg>', svgstr, re.DOTALL)
            if self.init_plot and len(self.win) < self.nb_of_automata:
                self.win.append(self.viz.svg(dot_g_svg.decode('utf-8')))
            else:
                self.init_plot = False
                self.viz.svg(dot_g_svg.decode('utf-8'), win=self.win[i])

