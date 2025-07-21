
from nltk.tree import Tree
import copy
import itertools
from numpy import insert
from collections import Counter

"""
Class to manage the transformation of a constituent tree into a sequence of labels
and vice versa. It extends the Tree class from the NLTK framework to address constituent Parsing as a 
sequential labeling problem.
"""
class SeqTree(Tree):
    
    EMPTY_LABEL = "EMPTY-LABEL"
    
    def __init__(self,label,children):
         
        self.encoding = None
        super(SeqTree, self).__init__(label,children) 
    
    #TODO: At the moment only the RelativeLevelTreeEncoder is supported
    def set_encoding(self, encoding):
        self.encoding = encoding

    """
    Transforms a constituent tree with N leaves into a sequence of N labels.
    @param is_binary: True if binary trees are being encoded and want to use an optimized
    encoding [Not tested at the moment]
    @param root_label: Set to true to include a special label to words directly attached to the root
    @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
    """
    def to_maxincommon_sequence(self,is_binary=False, root_label=False, encode_unary_leaf=False):
        
        if self.encoding is None: raise ValueError("encoding attribute is None")
        leaves_paths = []
        self.path_to_leaves([self.label()],leaves_paths)
        leaves = self.leaves()
        unary_sequence =  [s.label() for s in self.subtrees(lambda t: t.height() == 2)] #.split("+")
        return self.encoding.to_maxincommon_sequence(leaves, leaves_paths, unary_sequence, binarized=is_binary, 
                                                     root_label= root_label,
                                                     encode_unary_leaf=encode_unary_leaf)

    """
    Transforms a predicted sequence into a constituent tree
    @params sequence: A list of the predictions 
    @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
    @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
    concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
    """
    @classmethod
    def maxincommon_to_tree(cls, sequence, sentence, encoding):
        if encoding is None: raise ValueError("encoding parameter is None")
        return encoding.maxincommon_to_tree(sequence, sentence)


    """
    Gets the path from the root to each leaf node
    Returns: A list of lists with the sequence of non-terminals to reach each 
    terminal node
    """
    def path_to_leaves(self, current_path, paths):

        for i, child in enumerate(self):
            
            pathi = []
            if isinstance(child,Tree):
                common_path = copy.deepcopy(current_path)
                
                common_path.append(child.label()+"-"+str(i))
                child.path_to_leaves(common_path, paths)
            else:
                for element in current_path:
                    pathi.append(element)
                pathi.append(child)
                paths.append(pathi)
    
        return paths
    
    
    
"""
Encoder/Decoder class to transform a constituent tree into a sequence of labels by representing
how many levels in the tree there are in common between the word_i and word_(i+1) (in a relative scale) 
and the label (constituent) at that lowest ancestor.
"""
class RelativeLevelTreeEncoder(object):
        
    ROOT_LABEL = "ROOT"
    NONE_LABEL = "NONE"
    
    #TODO: The binarized option has not beend tested/evaluated
    """
    Transforms a tree into a sequence encoding the "deepest-in-common" phrase between words t and t+1
    @param leaves: A list of words representing each leaf node
    @param leaves_paths: A list of lists that encodes the path in the tree to reach each leaf node
    @param unary_sequence: A list of the unary sequences (if any) for every leaf node
    @param binarized: If True, when predicting an "ascending" level we map the tag to -1, as it is possible to determine in which
    level the word t needs to be located
    @param root_label: Set to true to include a special label ROOT to the words that are directly attached to the root of the sentence
    @param encode_unary_leaf: Set to true to encode leaf unary chains as a part of the label
    """  
    def to_maxincommon_sequence(self, leaves, leaves_paths, unary_sequence, 
                                binarized, root_label, encode_unary_leaf=False):
        
        sequence = []
        previous_ni = 0
        ni=0
        relative_ni = 0 
        
        for j,leaf in enumerate(leaves):
            
            #It is the last real word of the sentence
            if j == len(leaves)-1: 
                                
                if encode_unary_leaf and "+" in unary_sequence[j]:
                    encoded_unary_leaf = "_"+"+".join(unary_sequence[j].split("+")[:-1]) #The PoS tags is not encoded
                else:
                    encoded_unary_leaf = ""
 
#               #This corresponds to the implementation without the computation trick
#                sequence.append((self.NONE_LABEL+encoded_unary_leaf))
#                break

                #TODO: This is a computation trick that seemed to work better in the dev set
                #Sentences of length on are annotated with ROOT_UNARYCHAIN instead NONE_UNARYCHAIN                 
                if (root_label and len(leaves)==1):
                    sequence.append(self.ROOT_LABEL+encoded_unary_leaf)
                else:
                    sequence.append((self.NONE_LABEL+encoded_unary_leaf))
                break
            
            explore_up_to = min( len(leaves_paths[j]), len(leaves_paths[j+1]) )+1   
            ni = 0
            for i in range(explore_up_to):   
                     
                if leaves_paths[j][i] == leaves_paths[j+1][i]:
                    ni+=1
                else:              
                    relative_ni = ni - previous_ni              
                    if binarized:
                        relative_ni = relative_ni if relative_ni >=0 else -1
                        
                    if encode_unary_leaf and "+" in unary_sequence[j]:
                        encoded_unary_leaf = "_"+"+".join(unary_sequence[j].split("+")[:-1]) #The PoS tags is not encoded
                    else:
                        encoded_unary_leaf = ""
                    
                    if root_label and ni==1:
                        sequence.append(self.ROOT_LABEL+"_"+leaves_paths[j][ni-1]+encoded_unary_leaf)
                    else:
                        sequence.append(self._tag(relative_ni, leaves_paths[j][ni-1])+encoded_unary_leaf)
                    
                    previous_ni = ni
                    break

        return sequence    
    
        
    #TODO: It should be possible to remove this precondition
    """
    Uncollapses the INTERMEDIATE unary chains and also removes empty nodes that might be created when
    transforming a predicted sequence into a tree.
    @precondition: Uncollapsing/Removing-empty from the root must be have done prior to to call 
    this function
    """
    def uncollapse(self, tree):
        
        uncollapsed = []
        for child in tree:

            if type(child) == type(u'') or type(child) == type(""):
                uncollapsed.append(child)
            else:
                #It also removes EMPTY nodes
                while child.label() == SeqTree.EMPTY_LABEL and len(child) != 0:
                    child = child[-1]
                
                label = child.label()
                if '+' in label:
                     
                    label_split = label.split('+')
                    swap = Tree(label_split[0],[])

                    last_swap_level = swap
                    for unary in label_split[1:]:
                        last_swap_level.append(Tree(unary,[]))
                        last_swap_level = last_swap_level[-1]
                    last_swap_level.extend(child)
                    uncollapsed.append(self.uncollapse(swap))
                #We are uncollapsing the child node
                else:     
                    uncollapsed.append(self.uncollapse(child))
        
        tree = Tree(tree.label(),uncollapsed)
        return tree
    
    
    """
    Gets a list of the PoS tags from the tree
    @return A list containing the PoS tags
    """
    def get_postag_trees(self,tree):
        
        postags = []
        
        for nchild, child in enumerate(tree):
            
            if len(child) == 1 and type(child[-1]) == type(""):
                postags.append(child)
            else:
                postags.extend(self.get_postag_trees(child))
        
        return postags


    #TODO: The unary chain is not needed here.
    """
    Transforms a prediction of the form LEVEL_LABEL_[UNARY_CHAIN] into a tuple
    of the form (level,label):
    level is an integer or None (if the label is NONE or NONE_leafunarychain).
    label is the constituent at that level
    @return (level, label)
    """
    def preprocess_tags(self,pred):
        new_pred = []
        #///////////////////////Start Toqeer's CODE
        #print(pred)
        for i,index in enumerate(range(len(pred)-1,0,-1)):
            previous=index-1
            #print(index)
            splitter=pred[index].split('_')
            if len(splitter)==2:
                if pred[previous]=="NP" and splitter[1]!="PP-G": # wrongly predicted NPs
                    pred[previous] = str(int(splitter[0])+1) + "_NP"
                # Add 1 to next if label is PP-G and next is not PP-G or PP
                if (pred[previous]=="PP-G" or pred[previous]=="PP") and splitter[1]!="PP-G" and splitter[1]!="PP":
                    pred[previous]=str(int(splitter[0])+1)+"_"+pred[previous]
                # Add 1 to next number if functional labels are included
                if (len(pred[previous].split('-'))==2 and pred[previous].split('-')[0]=="PP") and splitter[1]!="PP-G" and splitter[1]!="PP":
                    pred[previous]=str(int(splitter[0])+1)+"_"+pred[previous]

                # If current is PP-G and next is also PP-G the both have same number
                if pred[previous]=="PP-G" and splitter[1]=="PP-G":
                    pred[previous] = str(int(splitter[0])) + "_" + pred[previous]
                # Add 1 to next if current lebel is PP or PP-G
                if pred[previous] == "PP-G" and splitter[1] == "PP":
                    pred[previous] = str(int(splitter[0])+1) + "_" + pred[previous]
                if pred[previous]=="PP" and splitter[1]=="PP":
                    pred[previous] = str(int(splitter[0])) + "_" + pred[previous]
                if pred[previous]=="PP" and splitter[1]=="PP-G":
                    pred[previous] = str(int(splitter[0])+1) + "_" + pred[previous]
                # if labels is NP and next is PP-G then add 1 to previous
                if pred[previous]=="NP" and splitter[1]=="PP-G":
                    pred[previous] = str(int(splitter[0]) + 1) + "_" + pred[previous]
                if pred[previous]=="NP" and splitter[1]=="NP":
                    pred[previous] = str(int(splitter[0])) + "_" + pred[previous]

                if pred[previous]=="DMP" and splitter[1]!="":
                    pred[previous] = str(int(splitter[0])+1) + "_" + pred[previous]
        # VC add 1 to previous so via reverse loop
        for i,index in enumerate(range(0,len(pred))): # forward loop for VC
            previous=index-1
            if pred[index] == "VC" and index == 0:
                splitter=pred[index+1].split('_')
                if len(splitter)==2:
                    pred[index]=str(int(splitter[0])+1)+"_VC"
            if pred[index] == "VC-VALA" and index == 0:
                splitter=pred[index+1].split('_')
                if len(splitter)==2:
                    pred[index]=str(int(splitter[0])+1)+"_VC-VALA"

            splitter=pred[previous].split('_')
            if len(splitter)==2:
                if pred[index]=="VC" and splitter[1]!="VC":
                    pred[index]=str(int(splitter[0])+1)+"_VC"
                if pred[index]=="VC" and splitter[1]=="VC":
                    pred[index]=str(int(splitter[0]))+"_VC"
                if pred[index]=="VC-VALA" and splitter[1]!="VC":
                    pred[index]=str(int(splitter[0])+1)+"_VC-VALA"
                if pred[index]=="VC-VALA" and splitter[1]=="VC":
                    pred[index]=str(int(splitter[0]))+"_VC-VALA"

        #outf.close()
        #print(pred) # To print all updated absolute labels
        #input()
        """
        #print("+++++++++++++++++++++++++++++++++++++++++")
        #print(pred)
        new_pred = []
        #print(pred)
        #print(len(pred))
        for i,lab in enumerate(pred):
            next=i+1
            next_next=next+1
            if (pred[i]=="PP-G" or pred[i]=="PP") and next<len(pred) and pred[next]!="NONE":
                splitter=pred[next].split("_")
                if len(splitter)>1:
                    num=int(splitter[0])+1
                    pred[i]=str(num)+"_"+pred[i]
            if (pred[i] == "PP" or pred[i] == "PP-G") and next < len(pred) and pred[next] != "NONE" and pred[next] == "PP":
                splitter = pred[next_next].split("_")
                if len(splitter) > 1:
                    num = int(splitter[0])
                    pred[i] = str(num) + "_" + pred[i]
                    pred[next] = str(num) + "_" + pred[next]
        for i,lab in enumerate(pred):
            next=i+1
            next_next=next+1
            if (pred[i]=="PP-G" or pred[i]=="PP") and next<len(pred) and pred[next]!="NONE":
                splitter=pred[next].split("_")
                if len(splitter)>1:
                    num=int(splitter[0])+1
                    pred[i]=str(num)+"_"+pred[i]
            if (pred[i] == "PP" or pred[i] == "PP-G") and next < len(pred) and pred[next] != "NONE" and pred[next] == "PP":
                splitter = pred[next_next].split("_")
                if len(splitter) > 1:
                    num = int(splitter[0])
                    pred[i] = str(num) + "_" + pred[i]
                    pred[next] = str(num) + "_" + pred[next]
        print(pred)
        """
        #input()
        #///////////////////// END Toqeer's CODE////////////////////////
        for label in pred:
            split=label.split("_")
            if len(split)==2 and split[0] not in ['ROOT','NONE']:
                new_pred.append((int(split[0]), split[1]))
            elif len(split)==2 and split[0] in ['ROOT']:
                new_pred.append((split[0], split[1]))
            elif len(split)==2 and split[0] in ['NONE']:
                new_pred.append((None, split[1]))
            elif len(split)==1 and split[0]=="NONE":
                new_pred.append((None, split[0]))
        #print(new_pred)
        return new_pred

        #print("inside preprocess tags function")
        #input()

        # try:
        #     label = pred.split("_")
        #     level, label = label[0],label[1]
        #     try:
        #         return (int(level), label)
        #     except ValueError:
        #
        #         #It is a NONE label with a leaf unary chain
        #         if level == self.NONE_LABEL: #or level == self.ROOT:
        #             return (None,pred.rsplit("_",1)[1])
        #
        #         return (level,label)
        #
        # except IndexError:
        #     #It is a NONE label (without any leaf unary chains)
        #     return (None, pred)
        
        

    """
    Transforms a predicted sequence into a constituent tree
    @params sequence: A list of the predictions 
    @params sentence: A list of (word,postag) representing the sentence (the postags must also encode the leaf unary chains)
    @precondition: The postag of the tuple (word,postag) must have been already preprocessed to encode leaf unary chains, 
    concatenated by the '+' symbol (e.g. UNARY[0]+UNARY[1]+postag)
    """
    def maxincommon_to_tree(self, sequence, sentence):
        #print(sentence)
        #print(sequence)


        tree = SeqTree(SeqTree.EMPTY_LABEL,[])
        current_level = tree
        previous_at = None
        first = True

        #print("---------------- CHecking MAP----------------------")
        #print(sequence)
        #input()
        sequence = self.preprocess_tags(sequence)
        #print(sequence)
        #input()

        sequence = self._to_absolute_encoding(sequence)
        #print(sequence)

        for j,(level,label) in enumerate(sequence):

            if level is None:
                prev_level, _ = sequence[j-1]
                previous_at = tree
                while prev_level > 1:
                    previous_at = previous_at[-1]
                    prev_level-=1
          
                #TODO: Trying optimitization
                #It is a NONE label
                if self.NONE_LABEL == label: #or self.ROOT_LABEL:
                #if "NONE" == label:
                    previous_at.append( Tree( sentence[j][1],[ sentence[j][0]]) )
                #It is a leaf unary chain
                else:
                    previous_at.append(Tree(label+"+"+sentence[j][1],[ sentence[j][0]]))   
                return tree
                continue
                   
            i=0
            #print(level-1)
            #input()
            for i in range(level-1):
                if len(current_level) == 0 or i >= sequence[j-1][0]-1: 
                    child_tree = Tree(SeqTree.EMPTY_LABEL,[])                      
                    current_level.append(child_tree)   
                    current_level = child_tree

                else:
                    current_level = current_level[-1]
                    
            if current_level.label() == SeqTree.EMPTY_LABEL:    
                current_level.set_label(label)
                        
            if first:
                previous_at = current_level
                previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))
                first=False
            else:         
                #If we are at the same or deeper level than in the previous step
                if i >= sequence[j-1][0]-1: 
                    current_level.append(Tree( sentence[j][1],[sentence[j][0]]))
                else:
                    previous_at.append(Tree( sentence[j][1],[ sentence[j][0]]))        
                previous_at = current_level
                
            current_level = tree
            
        return tree



    """
    Transforms an encoding of a tree in a relative scale into an
    encoding of the tree in an absolute scale.
    """
    def _to_absolute_encoding(self, relative_sequence):

        absolute_sequence = [0]*len(relative_sequence)
        current_level = 0
        for j,(level,phrase) in enumerate(relative_sequence):
            if level is None:
                absolute_sequence[j] = (level,phrase)
            elif level == self.ROOT_LABEL:
                absolute_sequence[j] = (1, phrase)
                current_level+=1
            else:
                current_level = level
                absolute_sequence[j] = (current_level,phrase)
        #print(absolute_sequence)
        #input()
        return absolute_sequence
    
    
    def _tag(self,level,tag):
        return str(level)+"_"+tag.rsplit("-",1)[0]
    



    
    
    
