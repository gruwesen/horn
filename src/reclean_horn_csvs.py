import argparse
import csv
import sys
from pathlib import Path

from sympy import And, Symbol, false, true


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import runFunctions


def _symbol_for(atom, symbols_by_atom):
    if atom not in symbols_by_atom:
        symbols_by_atom[atom] = Symbol(f"a{len(symbols_by_atom)}")
    return symbols_by_atom[atom]


def _parse_rule_sentence(sentence, symbols_by_atom):
    if sentence == "False":
        return false
    if sentence == "True":
        return true
    if " ---> " not in sentence:
        return _symbol_for(sentence, symbols_by_atom)

    antecedent_text, consequent_text = sentence.split(" ---> ", 1)
    antecedent_atoms = [part.strip() for part in antecedent_text.split(" & ") if part.strip()]
    antecedent_symbols = [_symbol_for(atom, symbols_by_atom) for atom in antecedent_atoms]
    antecedent = And(*antecedent_symbols) if antecedent_symbols else true
    consequent = false if consequent_text == "FALSE" else _symbol_for(consequent_text, symbols_by_atom)
    return antecedent >> consequent


def reclean_file(raw_file, clean_file, count_file):
    symbols_by_atom = {}
    sentence_by_rule = {}
    rules = []

    with raw_file.open(newline="", encoding="UTF-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            sentence = row[0]
            rule = _parse_rule_sentence(sentence, symbols_by_atom)
            rules.append(rule)
            sentence_by_rule.setdefault(rule, sentence)

    clean_rules = runFunctions._remove_entailed_rules(rules)
    clean_sentences = [[sentence_by_rule[rule]] for rule in clean_rules]

    clean_file.parent.mkdir(parents=True, exist_ok=True)
    count_file.parent.mkdir(parents=True, exist_ok=True)
    with clean_file.open("w", newline="", encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["EXTRACTED CLEAN HORN RULES"])
        writer.writerows(clean_sentences)

    with count_file.open("w", newline="", encoding="UTF-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["RAW_RULES", "CLEAN_RULES", "REMOVED_RULES"])
        writer.writerow([len(rules), len(clean_rules), len(rules) - len(clean_rules)])

    return len(rules), len(clean_rules)


def reclean_result_dir(result_dir):
    raw_dir = result_dir / "output_data" / "hornRules"
    clean_dir = result_dir / "output_data" / "hornRulesClean"
    count_dir = result_dir / "output_data" / "hornRulesCleanCounts"
    for raw_file in sorted(raw_dir.glob("*.csv")):
        raw_count, clean_count = reclean_file(
            raw_file,
            clean_dir / raw_file.name,
            count_dir / raw_file.name,
        )
        print(f"{raw_file.name}: {raw_count} -> {clean_count}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild cleaned Horn CSVs from output_data/hornRules")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    reclean_result_dir(args.result_dir)


if __name__ == "__main__":
    main()