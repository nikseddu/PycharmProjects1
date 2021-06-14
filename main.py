from Data_preprocessing import load_data_spacy
from Evaluate import  evaluate1
import spacy
import random
import time
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import sys
from spacy import displacy
from itertools import chain


# Getting Train ,validation and test Data and converting into the format Spacy takes


# Spacy Format
# train =[("Money Transfer from my checking account is not working",{"entities":[(6,13,"ACTIVITY"),(23,39,"PRODUCT")]}),
#         ("I want to check balance in saving account",{"entities":[(16,23,"ACTIVITY"),(30,45,"PRODUCT")]})
# ]


TRAIN_DATA, LABELS = load_data_spacy("Data/train.tsv")
# print(TRAIN_DATA)
print(len(TRAIN_DATA))
TEST_DATA, _ = load_data_spacy("Data/test.tsv")
print(len(TEST_DATA))
VALID_DATA, _ = load_data_spacy("Data/train_dev.tsv")
print(len(VALID_DATA))



def train_spacy(train_data, labels, iterations, dropout=0.5, display_freq=1):
   ''' Train a spacy NER model, which can be queried against with test data

   train_data : training data in the format of (sentence, {entities: [(start, end, label)]})
   labels : a list of unique annotations
   iterations : number of training iterations
   dropout : dropout proportion for training
   display_freq : number of epochs between logging losses to console
   '''
   valid_f1scores = []
   test_f1scores = []
   #nlp = spacy.load("en_core_web_md")
   nlp = spacy.blank('en')
   if 'ner' not in nlp.pipe_names:
       ner = nlp.create_pipe('ner')
       nlp.add_pipe('ner')
   else:
       ner = nlp.get_pipe("ner")

   # Add entity labels to the NER pipeline
   for i in labels:
       ner.add_label(i)

   # Disable other pipelines in SpaCy to only train NER
   other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
   with nlp.disable_pipes(*other_pipes):
       # nlp.vocab.vectors.name = 'spacy_model' # without this, spaCy throws an "unnamed" error
       optimizer = nlp.begin_training()
       for itr in range(iterations):
           random.shuffle(train_data)  # shuffle the training data before each iteration
           losses = {}
           for batch in minibatch(train_data, size=compounding(16.0, 64.0, 1.5)):
               for texts, annotations in batch:
                   doc = nlp.make_doc(texts)
                   example=Example.from_dict(doc,annotations)
                   nlp.update([example],drop=dropout,
                    sgd=optimizer,
                    losses=losses)




           # batches = minibatch(train_data, size=compounding(16.0, 64.0, 1.5))
           # for batch in batches:
           #     texts, annotations = zip(*batch)
           #     doc = nlp.make_doc(texts)
           #     example = Example.from_dict(doc, annotations)
               # nlp.update(
               #     texts,
               #     annotations,
               #     drop=dropout,
               #     sgd=optimizer,
               #     losses=losses)


           # if itr % display_freq == 0:
           #    print("Iteration {} Loss: {}".format(itr + 1, losses))
           scores = evaluate1(nlp, VALID_DATA)
           valid_f1scores.append(scores["textcat_f"])
           print('=======================================')
           print('Interation = ' + str(itr))
           print('Losses = ' + str(losses))
           print('===============VALID DATA========================')

           print('F1-score = ' + str(scores["textcat_f"]))
           print('Precision = ' + str(scores["textcat_p"]))
           print('Recall = ' + str(scores["textcat_r"]))
           scores = evaluate1(nlp, TEST_DATA)
           test_f1scores.append(scores["textcat_f"])
           print('===============TEST DATA========================')
           print('F1-score = ' + str(scores["textcat_f"]))
           print('Precision = ' + str(scores["textcat_p"]))
           print('Recall = ' + str(scores["textcat_r"]))
           print('=======================================')

   return nlp, valid_f1scores, test_f1scores


# Train (and save) the NER model
ner,valid_f1scores,test_f1scores = train_spacy(TRAIN_DATA, LABELS,20)
ner.to_disk("Model")


# #To get only NER pipeline
# ner = nlp.get_pipe('ner')
#
# #Adding ACTIVITY and PRODUCT as custom labels
# for _,annotation in train:
#     for ent in annotation.get("entities"):
#         ner.add_label(ent[2])
#
# #GEtting list of the all the other pipes available to disable them later
# disable_pipes = [pipe for pipe in nlp.pipe_names if pipe!='ner']
#
#
# with nlp.disable_pipes(*disable_pipes):
#     optimizer=nlp.resume_training()
#
# nlp.update(text)
