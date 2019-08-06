#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:16:21 2018

@author: joe
"""
import numpy as np

def backtraking(extremos, grupos, pos, num_grup, cont, size, nivel):
    
    ind = 0        

    while(cont < size):
        
        #por si acaso
        if pos == size:
            return extremos, cont

        #El primer nivel
        if grupos[pos] == -1 and nivel == 1:
            
            grupos[pos] = num_grup
            cont += 1

        #verificador de grupo
        if grupos[ind] == -1:
          
            if verificar_cercania(extremos,ind,pos):
                grupos[ind] = num_grup
                cont += 1
                grupos, cont = backtracking(extremos, grupos, ind, num_grup, cont, size, nivel + 1 )
        
        if ind < size:
            ind += 1

        if ind == size:

            if nivel != 1:
                return grupos, cont

            pos += 1
            ind = 0

            if nivel == 1 and grupos[pos] == -1:
                num_grup += 1
    return grupos, cont

def iniciar(extremos):
    
    size = len(extremos)
    
    check = np.zeros(size)

    grupos = [num-1 for num in np.zeros(size)]
    
    print grupos
    print check

    pos = 0    
    cont = 0
    num_grup = 0

    grupos, cont = backtracking(extremos, grupos, pos, num_grup, cont, size, 1)
