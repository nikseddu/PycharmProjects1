from itertools import chain
import numpy as np
import spacy
import random
import time
from spacy.util import minibatch, compounding
import sys
from spacy import displacy
from itertools import chain

"""
  Let us define methods to compute Precision,Recall and F1-score
  Model evaluation -> Dataset( train/test/validation)
  TP = Word predicted as either I_Disease or B_Disease and present in the data(train/test/validation) as as either I_Disease or B_Disease
  FP = Word predicted as either I_Disease or B_Disease and not present in the data (train/test/validation) as as either I_Disease or B_Disease
  FN = Word present in the data(train/test/validation) data as as either I_Disease or B_Disease but not predicted as as either I_Disease or B_Disease
  Metrics: Precision = TP/(TP+FP ) Recall = TP/(TP+FN) F1-score = 2 Precision * Recall/ (Precision+Recall)
  Precision : Total positive points divided  by Total Positive Points predicted by model
  Recall : Total Positive Points divided by Actual Total Positive points
  F1-Score : Harmonic Mean of Precision and Recall
"""


def calc_precision(pred, true):
   precision = len([x for x in pred if x in true]) / (len(pred) + 1e-20) # true positives / total pred
   return precision

def calc_recall(pred, true):
   recall = len([x for x in true if x in pred]) / (len(true) + 1e-20)    # true positives / total test
   return recall

def calc_f1(precision, recall):
   f1 = 2 * ((precision * recall) / (precision + recall + 1e-20))
   return f1




def evaluate1(ner, data ):
   preds = [ner(x[0]) for x in data]

   precisions, recalls, f1s = [], [], []

   # iterate over predictions and test data and calculate precision, recall, and F1-score
   for pred, true in zip(preds, data):
       true = [x[2] for x in list(chain.from_iterable(true[1].values()))] # x[2] = annotation, true[1] = (start, end, annot)
       pred = [i.label_ for i in pred.ents] # i.label_ = annotation label, pred.ents = list of annotations
       precision = calc_precision(true, pred)
       precisions.append(precision)
       recall = calc_recall(true, pred)
       recalls.append(recall)
       f1s.append(calc_f1(precision, recall))

   #print("Precision: {} \nRecall: {} \nF1-score: {}".format(np.around(np.mean(precisions), 3),np.around(np.mean(recalls), 3),
   #                                                         np.around(np.mean(f1s), 3)))
   return {"textcat_p": np.mean(precisions), "textcat_r": np.mean(recalls), "textcat_f":np.mean(f1s)}
