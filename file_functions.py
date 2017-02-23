# -*- coding: utf-8 -*-
"""
We store the weights of the tensorflow-networks via the tf.saver class. This creates a checkpoint.
To save time, we also store inside the checkpoint-file (which just contains the filenames of the actual checkpoint files) 
the number of certain iterations already done. As for that, we need the functions read_iteration and write_iteation.
Also, since we have an option to use a pre-trained word2vec (personally pre-trained), we want to have different 
checkpoints, one for the weights with the pre-trained wor2vec, and one without. If we switch the option, it shall decide
which checkpoint to use. For that, there is the function prepare_checkpoints.

@author: cstenkamp@uos.de
"""
import shutil
from pathlib import Path
import os


def read_iteration(string="Iteration", path="./"):
    '''
    Reads the number of iterations from the "checkpoint" file in the current folder.
    It goes through the file, and looks for the number after the argument-string
    '''
    try:
        with open(path+"checkpoint", encoding="utf8") as infile:
            for line in infile:
                line = line.replace("\n","")
                if line[:len(string)+2] == "#"+string+":":  #one line will be "#Iterations: " and then the number we seek
                     iterations = int(line[line.find('"'):][1:-1])
                     return iterations
            return 0 #if we didn't find the number, we assume it's zero.
    except FileNotFoundError:
        return 0  #if we didn't find the file at all, we assume it's zero.
    
    
    

def increase_iteration(string="Iteration", path="./"):
    '''
    Reads the number of iterations from the "checkpoint" file in the current folder
    and increases it by one.
    '''    
    write_iteration(read_iteration(string,path)+1,string,path)
    
    
    
    
def write_iteration(number, string="Iteration", path="./"):
    '''
    Writes the number of iterations, "number", inside the "checkpoint"-file
    '''   
    try:
        lines = []
        with open(path+"checkpoint", encoding="utf8") as infile:
            for line in infile: #we read the file, and simply copy every line of it into the array 'lines'...
                line = line.replace("\n","")
                if not line[:len(string)+2] == "#"+string+":": #...except for the one containing the previous number of iterations...
                    lines.append(line) 
            lines.append("#"+string+': "'+str(number)+'"') #..that one is replaced by the argument number.
    except FileNotFoundError:
        lines.append("#"+string+': "'+str(number)+'"') #if the checkpoint file doesn't exist at all, at least write the number of iterations.
    infile = open(path+"checkpoint", "w") #create a new file in writing mode,
    infile.write("\n".join(lines));  #and dump the content of our "lines" into it.
    infile.close()     
    
    
    
    

def prepare_checkpoint(usew2v, path="./"):
    '''
    Makes a backup from the current checkpoint-file in a backup-file (which backup-file depends on if we used the pretrained word2vec or not..)
    And then re-creates the checkpoint-file from the appropriate backup (which one depends on if we want to use the pretrained word2vec or not)
    '''       
    def check_which(path="./"):
        '''
        Checks if the current checkpoint was one using word2vec or not:
        Returns 1 if it was, 2 if it wasn't, and 0 if there is no current checkpoint
        '''
        try:
            with open(path+"checkpoint", encoding="utf8") as infile:
                for line in infile:
                    if line.find("_wordvecs") > 0:
                        return 1
                return 2
        except FileNotFoundError:
            return 0
        
    #backup the current checkpoint...
    if check_which(path) == 1:
        shutil.copy(path+"checkpoint",path+".checkpointbkp_withwordvecs")
    elif check_which(path) == 2:
        shutil.copy(path+"checkpoint",path+".checkpointbkp_nowordvecs")
    else:
        return
    
    #if the current checkpoint is not appropriate for the use_w2v-setting, load a backup if available.
    if not ((check_which(path) == 1 and usew2v) or (check_which(path) == 2 and not usew2v)): #only need to do this if we switched from not-using to using or vice versa
        if usew2v:
            if Path(path+".checkpointbkp_withwordvecs").is_file():
                shutil.copy(path+".checkpointbkp_withwordvecs",path+"checkpoint")
            else:
                print("No previous checkpoint found, deleting the old one!")
                os.remove(path+"checkpoint")
                
        else:
            if Path(path+".checkpointbkp_nowordvecs").is_file():
                shutil.copy(path+".checkpointbkp_nowordvecs",path+"checkpoint")
            else:
                print("No previous checkpoint found, deleting the old one!")
                os.remove(path+"checkpoint")