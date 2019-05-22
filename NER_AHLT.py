#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import pandas as pd
import re

# List of drugs from The Drug Ontology (BioPortal) (https://bioportal.bioontology.org/ontologies/DRON)
drugs_dron = 'dron_clean.csv'
dron_df = pd.read_csv(drugs_dron, header=None, sep='#')
list_dron = dron_df.values.tolist()
# list of list of numbers to list of numbers
ldron = [item for sublist in list_dron for item in sublist]
dron = []
nope = ['Preferred', 'Label', '/', '%', 'ML', 'MG', 'MG/MG', 'MG/ML', 'ML/ML', 'ML/MG', 'UNT', 'UNT/ML', 'UNT/MG',
        'Oral', 'Tablet', 'MG/HR', 'NX']
for d in ldron:
  if d not in nope:
    d = str(d).strip('[')
    d = d.strip(']')
    d = d.strip('(')
    d = d.strip(')')
    dron.append(d.lower())

# --------- tokenize sentence -----------
# -- Tokenize sentence, returning tokens and span offsets


def tokenize(txt):
  offset = 0
  tks = []
  # word_tokenize splits words, taking into account punctuations, numbers, etc.
  for t in word_tokenize(txt):
    # keep track of the position where each token should appear, and
    # store that information with the token
    offset = txt.find(t, offset)
    tks.append((t, offset, offset + len(t) - 1))
    offset += len(t)

  # tks is a list of triples (word,start,end)
  return tks


# --------- get tag -----------
# Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans):
  (form, start, end) = token
  for (spanS, spanE, spanT) in spans:
    if start == spanS and end <= spanE:
      return "B-" + spanT
    elif start >= spanS and end <= spanE:
      return "I-" + spanT

  return "O"

# --------- Feature extractor -----------
# -- Extract features for each token in given sentence


def extract_features(tokens):

  # for each token, generate list of features and add it to the result
  result = []

  #  POST-Tag i Lemmas
  # per a cada tupla ('Co-administration', 0, 16)... agafem el token[x][0] que serà la paraula
  letters = [tokens[k][0] for k in range(0, len(tokens))]
  #print('letters: {}'.format(letters))
  pairs = nltk.pos_tag(letters)
  #print('pairs: {}'.format(pairs))
  lpostags = [p[1] for p in pairs]
  llemmas = [lemmatize(pair) for pair in pairs]

  for k in range(0, len(tokens)):
    tokenFeatures = []
    t = tokens[k][0]

    tokenFeatures.append("form=" + t)
    tokenFeatures.append("formlower=" + t.lower())
    tokenFeatures.append("suf3=" + t[-3:])
    tokenFeatures.append("suf4=" + t[-4:])
    tokenFeatures.append("length:" + str(len(t)))  # length of token
    if (t.isupper()):
      tokenFeatures.append("isUpper")
    if (t.istitle()):
      tokenFeatures.append("isTitle")
    if (t.isdigit()):
      tokenFeatures.append("isDigit")
    # has + or - or , or number inside the word
    if (re.match(r'[a-zA-Z0-9]*(\+|-|,|\(|\)|[0-9])[a-zA-Z0-9]*', t)):
      tokenFeatures.append("hasSymbol")
    if (t[-1:].isdigit()):
      tokenFeatures.append("lastDigit")   # last character is digit

    if (t.lower() in dron):
      tokenFeatures.append("inDron")  # in Dron list

    if k > 0:
      tPrev = tokens[k - 1][0]
      tokenFeatures.append("formPrev=" + tPrev)
      tokenFeatures.append("formlowerPrev=" + tPrev.lower())
      tokenFeatures.append("suf3Prev=" + tPrev[-3:])
      tokenFeatures.append("suf4Prev=" + tPrev[-4:])
      tokenFeatures.append("lengthPrev:" + str(len(tPrev)))
      if (t.isupper()):
        tokenFeatures.append("isUpperPrev")
      if (t.istitle()):
        tokenFeatures.append("isTitlePrev")
      if (t.isdigit()):
        tokenFeatures.append("isDigitPrev")
      if (re.match(r'[a-zA-Z0-9]*(\+|-|,|\(|\)|[0-9])[a-zA-Z0-9]*', tPrev)):
        tokenFeatures.append("hasSymbolPrev")
      if (tPrev[-1:].isdigit()):
        tokenFeatures.append("lastDigitPrev")
      if (tPrev.lower() in dron):
        tokenFeatures.append("inDronPrev")

    else:
      tokenFeatures.append("BoS")

    if k < len(tokens) - 1:
      tNext = tokens[k + 1][0]
      tokenFeatures.append("formNext=" + tNext)
      tokenFeatures.append("formlowerNext=" + tNext.lower())
      tokenFeatures.append("suf3Next=" + tNext[-3:])
      tokenFeatures.append("suf4Next=" + tNext[-4:])
      tokenFeatures.append("lengthNext:" + str(len(tNext)))
      if (t.isupper()):
        tokenFeatures.append("isUpperNext")
      if (t.istitle()):
        tokenFeatures.append("isTitleNext")
      if (t.isdigit()):
        tokenFeatures.append("isDigitNext")
      if (re.match(r'[a-zA-Z0-9]*(\+|-|,|\(|\)|[0-9])[a-zA-Z0-9]*', tNext)):
        tokenFeatures.append("hasSymbolNext")
      if (tNext[-1:].isdigit()):
        tokenFeatures.append("lastDigitNext")
      if (tNext.lower() in dron):
        tokenFeatures.append("inDronNext")

    else:
      tokenFeatures.append("EoS")

    result.append(tokenFeatures)

  return result


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir, and writes
# -- them in the output format requested by the evalution programs.
# --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

  # parse XML file, obtaining a DOM tree
  tree = parse(datadir + "/" + f)

  # process each sentence in the file
  sentences = tree.getElementsByTagName("sentence")
  for s in sentences:
    sid = s.attributes["id"].value   # get sentence id
    spans = []
    stext = s.attributes["text"].value   # get sentence text
    entities = s.getElementsByTagName("entity")
    for e in entities:
      # for discontinuous entities, we only get the first span
      # (will not work, but there are few of them)
      (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
      typ = e.attributes["type"].value
      spans.append((int(start), int(end), typ))

    # convert the sentence to a list of tokens
    tokens = tokenize(stext)
    # extract sentence features
    features = extract_features(tokens)

    # print features in format expected by crfsuite trainer
    for i in range(0, len(tokens)):
      # see if the token is part of an entity
      tag = get_tag(tokens[i], spans)
      print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

    # blank line to separate sentences
    print()
