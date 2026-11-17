# Adapted from the HORN implementation of Blum, Koudijs, Ozaki and Touileb
# (Int. J. Approx. Reason. 171 (2024) 109026); code base https://github.com/ZofieB/MT_UiB.
# See README.md.
import numpy as np
from itertools import combinations
from sympy import symbols
import csv
import datetime
import os
from sympy.logic.boolalg import And, Not
from sympy.logic.inference import satisfiable


def rulesToSentences(rules, lookupTable):
    lookupTable = sum([x[0] for x in lookupTable.values()],[]) # removing excess metadata from the full lookuptable
    strRules = list(map(lambda x: str(x),rules)) # symbol rules to string rules
    ruleSentences = []
    for rule in strRules:
        # print(rule)
        match rule[0]: # matching on the first letter in string
            case "I": # Case implication                                                        # rule = Implies(v10 & v20 & v23 & v28 & v5, v19)
                antRules = rule[8:-1].split(", ")[0].split(" & ")                               # [v10, v20, v23, v28, v5]
                conRule = rule[8:-1].split(", ")[1]                                             # [v11]
                antSentence = " & ".join(list(map(lambda x: lookupTable[int(x[1:])],antRules))) # mekaniker & Bodø & Afrika & kvinne & eldre enn 60
                conSentence = lookupTable[int(conRule[1:])]                                     # elektriker
                ruleSentence = f"{antSentence} ---> {conSentence}"                              # mekaniker & Bodø & Afrika & kvinne & eldre enn 60 ---> elektriker
                ruleSentences.append([ruleSentence])
            case "~": #case negation
                statements = rule[2:-1].split(" & ")
                if len(statements) == 1: # if the negation clause only contains one value
                    ruleSentences.append([f"{lookupTable[int(rule[2:])]} ---> FALSE"])
                else:
                    statementsSentences = " & ".join(list(map(lambda x: lookupTable[int(x[1:])],statements)))
                    ruleSentences.append([f"{statementsSentences} ---> FALSE"])
            case "F": # False
                ruleSentences.append(["False"])
            case "T": # True
                ruleSentences.append(["True"])
            case "v": # single rule
                ruleSentences.append([lookupTable[int(rule[1:])]])
            case default: #case true????????
                ruleSentences.append([f"default = {rule}"])
                # print(f"default = {rule}")
    ruleSentences.sort(key=lambda x: x[0])# sorting
    return ruleSentences

# pre Horn Algorithm functions ########################################################################
def define_variables(number):
    s = "".join(['v'+str(i)+',' for i in range(number)])
    V = [e for e in symbols(s)]
    return V

def generateBackground(V, attLengths):
    # two values from the same attrutube dimention cant be true simultaneously
    splitIndexes = []
    i = 0
    for x in attLengths:
        i += x
        splitIndexes.append(i)
    splitted = np.split(V, splitIndexes)
    background = set()

    for x in splitted:
        background.update(set(combinations(x, 2)))

    background = set(map(lambda x: ~(x[0] & x[1]),background))

    #storing background
    return background

def storeBackground(background, lookupTable):
    # strBackground = list(map(lambda x: str(x),background))
    backgroundSentences = rulesToSentences(background,lookupTable)
    with open('output_data/backgroundRules/backgroundRules.csv', 'w', newline = '', encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([["BACKGROUND"]])
        writer.writerows(backgroundSentences)

# post Horn Algorithm functions #######################################################################
def storeMetadata(writeTo, metadata):
    newMetadata = [] # adding total time to metadata

    totalRuntime = 0
    for x in metadata:
        totalRuntime += x[-1]
        newMetadata.append(x + [float(f"{totalRuntime:.3}")] + [str(datetime.timedelta(seconds=int(totalRuntime)))])

    with open(f"output_data/metadata/{writeTo}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerows([["ITERATION", "LEN(HYP)", "SAMPLENR", "RUNTIME(sample)", "TOTALRUNTIME(seconds)","TOTALRUNTIME(time)"]]) #header
        writer.writerows(newMetadata)


def storeHornRules(writeTo, h, lookupTable):
    os.makedirs("output_data/hornRules", exist_ok=True)
    os.makedirs("output_data/hornRulesClean", exist_ok=True)
    os.makedirs("output_data/hornRulesCleanCounts", exist_ok=True)

    hornRuleSentences = rulesToSentences(h,lookupTable)
    with open(f"output_data/hornRules/{writeTo}.csv", 'w', newline='',encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["EXTRACTED HORN RULES"])
        writer.writerows(hornRuleSentences)

    clean_rules = _remove_entailed_rules(list(h))
    cleanHornRuleSentences = rulesToSentences(clean_rules, lookupTable)
    with open(f"output_data/hornRulesClean/{writeTo}.csv", 'w', newline='', encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["EXTRACTED CLEAN HORN RULES"])
        writer.writerows(cleanHornRuleSentences)

    raw_count = len(hornRuleSentences)
    clean_count = len(cleanHornRuleSentences)
    removed_count = raw_count - clean_count
    with open(f"output_data/hornRulesCleanCounts/{writeTo}.csv", 'w', newline='', encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["RAW_RULES", "CLEAN_RULES", "REMOVED_RULES"])
        writer.writerow([raw_count, clean_count, removed_count])

    return raw_count, clean_count


def _remove_entailed_rules(rules):
    """
    Greedily remove rules entailed by the rules kept so far.

    A rule set may contain cycles where each rule is entailed by the original
    rest of the set.  Testing every rule against the unmodified original set can
    delete the whole cycle, so cleanup must be incremental.  Contradiction rules
    must be considered first: a -> FALSE entails every a -> x rule.
    """
    kept = []
    for rule in sorted(rules, key=_cleanup_sort_key):
        if not _is_entailed(rule, kept):
            kept.append(rule)
    return kept


def _cleanup_sort_key(rule):
    antecedent, consequent = _split_implication(rule)
    consequent_priority = 0 if consequent == False else 1
    antecedent_size = len(getattr(antecedent, "free_symbols", ()))
    return (consequent_priority, antecedent_size, str(antecedent), str(consequent), str(rule))


def _split_implication(rule):
    if getattr(rule, "func", None).__name__ == "Implies" and len(rule.args) == 2:
        return rule.args
    if getattr(rule, "func", None).__name__ == "Not" and len(rule.args) == 1:
        return (rule.args[0], False)
    return (True, rule)


def _is_entailed(rule, others):
    premise = And(*others) if others else True
    check_expr = And(premise, Not(rule))
    sat_result = satisfiable(check_expr)
    return sat_result is False
