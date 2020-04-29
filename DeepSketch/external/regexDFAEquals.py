import sys
import argparse
import subprocess

def dfa_eual_test(gold, predicted):
    gold = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    return regex_equiv(gold, predicted)

def silent_eual_test(pair):
    gold, predicted = pair
    gold = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    return silent_regex_equiv(gold, predicted)

def unprocess_regex(regex):
    # regex = regex.replace("<VOW>", " ".join('[AEIOUaeiou]'))
    # regex = regex.replace("<NUM>", " ".join('[0-9]'))
    # regex = regex.replace("<LET>", " ".join('[A-Za-z]'))
    # regex = regex.replace("<CAP>", " ".join('[A-Z]'))
    # regex = regex.replace("<LOW>", " ".join('[a-z]'))

    regex = regex.replace("<VOW>", " ".join('AEIOUaeiou'))
    regex = regex.replace("<NUM>", " ".join('0-9'))
    regex = regex.replace("<LET>", " ".join('A-Za-z'))
    regex = regex.replace("<CAP>", " ".join('A-Z'))
    regex = regex.replace("<LOW>", " ".join('a-z'))

    regex = regex.replace("<M0>", " ".join('dog'))
    regex = regex.replace("<M1>", " ".join('truck'))
    regex = regex.replace("<M2>", " ".join('ring'))
    regex = regex.replace("<M3>", " ".join('lake'))

    regex = regex.replace(" ", "")

    return regex

def regex_equiv(gold, predicted):
    print("GOLD", gold)
    print("PRED", predicted)
    if gold == predicted:
        print("PERFECT")
        return True
    try:
        out = subprocess.check_output(
            ['java', '-jar', './external/regex_dfa_equals.jar', '{}'.format(gold), '{}'.format(predicted)])
        print("OUT", out)
        if '\\n1' in str(out):
            return True
        else:
            return False
    except Exception as e:
        return False
    return False

def silent_regex_equiv(gold, predicted):
    if gold == predicted:
        return "perfect"
    try:
        out = subprocess.check_output(
            ['java', '-jar', './external/regex_dfa_equals.jar', '{}'.format(gold), '{}'.format(predicted)])
        if '\\n1' in str(out):
            return "true"
        else:
            return "false"
    except Exception as e:
        return "false"
    return "false"

def regex_equiv_from_raw(gold, predicted):
    gold = unprocess_regex(gold)
    predicted = unprocess_regex(predicted)
    return regex_equiv(gold, predicted)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
