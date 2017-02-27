# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:32:12 2017

@author: csten_000
"""

import os
import copy

def line_count(filename):
    """
    Count file's lines.
    http://stackoverflow.com/a/27518377/1447384
    """
    def _line_count_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024*1024)
    f = open(filename, 'rb')
    f_gen = _line_count_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)



def preparestring(string):
    str = copy.deepcopy(string)
    str = str.lower()
    str = str.replace(",", " <comma> ")
    str = str.replace(":", " <colon> ")
    str = str.replace("(", " <openBracket> ")
    str = str.replace(")", " <closeBracket> ")
    str = str.replace("...", " <dots> ")
    str = str.replace(".", " <dot> ")
    str = str.replace(";", " <semicolon> ")
    str = str.replace('"', " <quote> ")
    str = str.replace("?", " <question> ")
    str = str.replace("!", " <exclamation> ")
    str = str.replace("-", " <hyphen> ")
    str = str.replace("???", " <SuperQuestion> ")
    str = str.replace("!!!", " <SuperExclamation> ")
    while str.find("  ") > 0: str = str.replace("  "," ")
    if str.endswith(' '): str = str[:-1]
    return str


def correct_grammar(string):
    str = copy.deepcopy(string)
    str = str.replace("'ve", " have")
    str = str.replace("w/o", "without")
    str = str.replace("w/", "with")
    str = str.replace("'s", " is")
    str = str.replace("'m", " am")
    str = str.replace("n't", " not")
    return str
    


def preprocess(text):
    text = correct_grammar(text)
    text = preparestring(text)
    return text



def create_from_johannes(frompath, positive="Filtered Tweets positive.txt", negative="Filtered Tweets negative.txt", amount_train=30000, intopath="./trumpsets/"):
    
    amount_test = amount_valid = amount_train//4
    files = ["train","test","validation"]
    target_appendix = "-target"    

    assert line_count(frompath+positive) >= (amount_test + amount_valid + amount_train)/2
    assert line_count(frompath+negative) >= (amount_test + amount_valid + amount_train)/2
    counter = 0
    sets = [[],[],[]]
    target = [[],[],[]]
    for openfile in [frompath+positive, frompath+negative]:
        with open("./"+openfile, encoding="utf8") as infile:
            for line in infile: 
                currtweet = preprocess(line)
                if counter < amount_train:
                    sets[0].append(currtweet)
                    target[0].append("1" if openfile == frompath+positive else "0")
                elif counter < amount_train + amount_test:
                    sets[1].append(currtweet)
                    target[1].append("1" if openfile == frompath+positive else "0")
                elif counter < amount_train + amount_test + amount_valid:
                    sets[2].append(currtweet)
                    target[2].append("1" if openfile == frompath+positive else "0")
                else:
                    break
                counter += 1
        counter = 0
    if not os.path.exists(intopath):
        os.makedirs(intopath)        
    thefiles = [i+".txt" for i in files] + [i+target_appendix+".txt" for i in files]
    i = 0
    for savefile in thefiles:
        infile = open(intopath+savefile, "w")
        if i < 3:
            infile.write("".join(sets[i]))
        else:
            infile.write("\n".join(target[i-3]))
        infile.close()     
        i += 1
        
        
if __name__ == '__main__':
    create_from_johannes(frompath="./")