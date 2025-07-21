# -*- coding: utf-8 -*-

from __future__ import print_function
from nltk import tree
from zipfile import ZipFile
from collections import Counter
import itertools
from nltk import Tree

import time
from tree2 import SeqTree, RelativeLevelTreeEncoder


def get_enriched_labels_for_retagger(preds, unary_preds):

    new_preds = []
    for zpreds, zunaries in zip(preds, unary_preds):
        aux = []
        for zpred, zunary in zip(zpreds, zunaries):
            if "+" in zunary and zpred not in ["-EOS-", "NONE", "-BOS-"]:

                if zpred == "ROOT":
                    new_zpred = "+".join(zunary.split("+")[:-1])
                else:
                    new_zpred = zpred + "_" + "+".join(zunary.split("+")[:-1])
            else:
                new_zpred = zpred
            aux.append(new_zpred)
        new_preds.append(aux)
    #print(new_preds)
    return new_preds

def sequence_to_parenthesis(sentences, labels):

    parenthesized_trees = []
    relative_encoder = RelativeLevelTreeEncoder()
    #print(relative_encoder)
    #input()
    f_max_in_common = SeqTree.maxincommon_to_tree
    f_uncollapse = relative_encoder.uncollapse

    total_posprocessing_time = 0
    for noutput, output in enumerate(labels):
        if output != "":  # We reached the end-of-file
            init_parenthesized_time = time.time()
            sentence = []
            preds = []
            for ((word, postag), pred) in zip(sentences[noutput][1:-1], output[1:-1]):

                if len(pred.split("_")) == 3:  # and "+" in pred.split("_")[2]:
                    sentence.append((word, pred.split("_")[2] + "+" + postag))

                else:
                    sentence.append((word, postag))

                    # TODO: This is currently needed as a workaround for the retagging strategy and sentences of length one
                #                 if len(output)==3 and output[1] == "ROOT":
                #                     pred = "NONE"

                preds.append(pred)
            #print("================================")
            #print(sentence)
            #print(preds)
            #input()
            tree = f_max_in_common(preds, sentence, relative_encoder)

            # Removing empty label from root
            if tree.label() == SeqTree.EMPTY_LABEL:

                # If a node has more than two children
                # it means that the constituent should have been filled.
                if len(tree) > 1:
                    print
                    "WARNING: ROOT empty node with more than one child"
                else:
                    while (tree.label() == SeqTree.EMPTY_LABEL) and len(tree) == 1:
                        tree = tree[0]

            # Uncollapsing the root. Rare needed
            if "+" in tree.label():
                aux = SeqTree(tree.label().split("+")[0], [])
                aux.append(SeqTree("+".join(tree.label().split("+")[1:]), tree))
                tree = aux
            tree = f_uncollapse(tree)

            total_posprocessing_time += time.time() - init_parenthesized_time
            # To avoid problems when dumping the parenthesized tree to a file
            aux = tree.pformat(margin=100000000)
            parenthesized_trees.append(aux)
        #print(parenthesized_trees[0])
    return parenthesized_trees

#####################################################################################################
#                       LABELS 2 BRACKETS
#####################################################################################################

print('Loading labels file...')

f = open("trained_models/outputs/output.seq_r", "r")
label =f.read()
_labels = [[row.split() for row in sample.split('\n')] for sample in label.strip().split('\n\n')]

#print(_labels[0])

sentences=[]
pos=[]
labels=[]
for sents in _labels:
    word_line=[]
    pos_line = []
    lab_line=[]
    for words in sents:
        word_line.append((words[0],words[1]))
        pos_line.append((words[1]))
        lab_line.append(words[2])
    sentences.append(word_line)
    pos.append(pos_line)
    labels.append(lab_line)
#print(sentences[0])
#print(labels[0])
#print("-------------------------------------------------")

#new_labels = get_enriched_labels_for_retagger(labels, pos)
brackets = sequence_to_parenthesis(sentences,labels)

outf = open("trained_models/outputs/output.brackets", "w")
for bracket in brackets:
    outf.write(bracket+"\n")
outf.close()

print("Bracketed trees have been written on disk...")
#import os
#os.system("/home/toqeer/PycharmProjects/urdu-parser/EVALB/evalb urdu_labels/function_labels_final/outputs/ref-fun.txt urdu_labels/function_labels_final/outputs/output_2_gpos_final.brackets -p /home/toqeer/PycharmProjects/urdu-parser/EVALB/new.prm > urdu_labels/function_labels_final/outputs/--final-fun_gpos.txt")
#os.system("./EVALB/evalb EVALB/results/ref_pos2.txt EVALB/results/brackets.txt -p EVALB/new.prm")
