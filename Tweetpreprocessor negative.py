# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 00:54:53 2017

@author: Johannes
"""
import re
                
def read_to_string(name):
    try:
        array = []
        with open(name, encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if len(line) > 0: array.append(line)
            return array
    except FileNotFoundError:
        return []


def mergetxt(namelist):
    alltxt = open('anti text.txt', 'a')
    merginglist = read_to_string(namelist+'.txt')
    for f in merginglist:
        with open (f+'.txt') as infile:
            alltxt.write(infile.read())
    filterforcontent('anti text')
            
def filterforcontent(alltwts):
    try:
        outfile = open('Filtered Tweets negative.txt','a')
        with open(alltwts+'.txt', encoding="utf8") as fltfile:
            for line in fltfile:
                if not 'http' in line:
                    if len(line) > 50:
                        while line.find("  ") > 0:
                            line = line.replace("  ","")
                        line = line[1:-1]
                        line = re.sub('(\\\\u\\w+?)+?\\b','',line)
                        line = line.replace('\\','')
                        line = line.replace('RT ','')
                        line = line.replace('#','')
                        line = line.replace('&amp','and')
                        line = line.replace('w/a','without')
                        line = line.replace('w/', 'with')
                        line = line[:-1]
                        outfile.write(line + '\n')
    except FileNotFoundError:
        return []
    file_len('Filtered tweets negative.txt')

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    print(i + 1)          
                
        

mergetxt('negativeaccountlist')
        
        



            
def read_iteration():
    try:
        with open("checkpoint", encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if line[:11] == "#Iteration:":
                     iterations = int(line[line.find('"'):][1:-1])
                     return iterations
            return 0
    except FileNotFoundError:
        return 0

def increase_iteration():
    write_iteration(read_iteration()+1)
    
def write_iteration(number):
    try:
        lines = []
        with open("checkpoint", encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if not line[:11] == "#Iteration:":
                    lines.append(line)
            lines.append('#Iteration: "'+str(number)+'"')
    except FileNotFoundError:
        lines.append('#Iteration: "'+str(number)+'"')
    infile = open("checkpoint", "w")
    infile.write("\n".join(lines));        
    infile.close()
                
        
    

    