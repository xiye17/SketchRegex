import json
import os
import sys
import shutil
import pickle
import subprocess
from data import get_cache_file
from external.regexDFAEquals import unprocess_regex, silent_eual_test
import random
import time

class DFAWorker:
    def __init__(self):
        pass

    def run(self, pair):
        gold, predicted = pair
        gold = unprocess_regex(gold)
        predicted = unprocess_regex(predicted)
        if gold == predicted:
            return "true"
        try:
            out = subprocess.check_output(
                ['java', '-jar', './external/regex_dfa_equals.jar', '{}'.format(gold), '{}'.format(predicted)], timeout=2)
            if '\\n1' in str(out):
                return "true"
            else:
                return "false"
        except Exception:
            return "false"
        return "false"

class SynthWorker:
    def __init__(self, dataset, split):
        self.split = split
        if dataset.startswith("TurkSketch"):
            self.timeout = 2
            self.mode = "1"
        if dataset.startswith("KB13Sketch"):
            self.timeout = 4
            self.mode = "2"

    def run(self, sketch):
        # java -Djava.library.path=external/lib -cp external/resnax.jar:external/lib/* -ea resnax.Main 1 $1 $2 $3
        cmd = ["java", "-Djava.library.path=external/lib", "-cp", "external/resnax.jar:external/lib/*", "-ea", "resnax.Main", self.mode, self.split, str(sketch[0]), sketch[1]]
        try:
            # out = str(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=self.timeout))
            out = str(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=self.timeout))
            if "true" in out:
                result = "true"
            elif "false" in out:
                result = "false"
            elif "wrong" in out:
                result = "wrong"
            elif "null" in out:
                result = "null"
            elif "empty" in out:
                result = "empty"
        except subprocess.TimeoutExpired:
            result = "timeout"
        except subprocess.CalledProcessError:
            result = "wrong"
        except ValueError:
            result = "wrong"

        return result

    def timed_run(self, sketch):
        t_start = time.monotonic()
        result = self.run(sketch)
        return result, (time.monotonic() - t_start)

class DFACache(object):
    def __init__(self, cache_id, dataset):
        self.cache_id = "DFA-" + dataset + '-' + cache_id
        self.cache_file = get_cache_file(self.cache_id)
        print(self.cache_file)
        self.data = {}

        self.load()
    
    def load(self):
        if not os.path.isfile(self.cache_file):
            self.data = {}
        else:
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print("Load {} recored".format(len(self.data)))

    def query(self, r1, r2):
        key = r1 + "DFADIV" + r2
        if key in self.data:
            return self.data[key]
        else:
            result = silent_eual_test((r1, r2))
            self.data[key] = result
            return result
    
    def rewrite(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.data, f)

    def soft_query(self, r1, r2):

        key = r1 + "DFADIV" + r2
        if key in self.data:
            return self.data[key]
        else:
            None
    
    def soft_write(self, r1, r2, result):
        key = r1 + "DFADIV" + r2
        self.data[key] = result

class SynthCache(object):

    def __init__(self, cache_id, dataset):
        self.dataset = dataset
        if dataset.startswith("TurkSketch"):
            self.cache_id = "Turk-" + cache_id
            self.cache_file = get_cache_file(self.cache_id)
            self.timeout = 2
            self.mode = "1"
        if dataset.startswith("KB13Sketch"):
            self.cache_id = "KB-" + cache_id
            self.cache_file = get_cache_file(self.cache_id)
            self.timeout = 5
            self.mode = "2"

        self.data = {}
        self.load()

    def load(self):
        if not os.path.isfile(self.cache_file):
            self.data = {}
        else:
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print("Load {} recored".format(len(self.data)))

    def rewrite(self):
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)

    def run_synth(self, split, id, sketch):

        # java -Djava.library.path=external/lib -cp external/resnax.jar:external/lib/* -ea resnax.Main 1 $1 $2 $3
        cmd = ["java", "-Djava.library.path=external/lib", "-cp", "external/resnax.jar:external/lib/*", "-ea", "resnax.Main", self.mode, split, str(id), sketch]
        try:
            out = str(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=self.timeout))
            if "true" in out:
                result = "true"
            elif "false" in out:
                result = "false"
            elif "wrong" in out:
                result = "wrong"
            elif "null" in out:
                result = "null"
            elif "empty" in out:
                result = "empty"
        except subprocess.TimeoutExpired:
            result = "timeout"
        except subprocess.CalledProcessError:
            result = "wrong"
        except ValueError:
            result = "wrong"
            print("Value Error", split, id, sketch)
            print("Value Error!!!!!!", split, id, sketch, file=sys.stderr)

        return result

    def clean_split(self, split):
        for k in list(self.data.keys()):
            if k.startswith(split):
                del self.data[k]

    def query(self, split, id, sketch):

        key = split + str(id)
        if key in self.data:
            vars = self.data[key]
            if sketch in vars:
                # get result
                return vars[sketch]
            else:
                result = self.run_synth(split, id, sketch)
                self.data[key][sketch] = result
                return result
        else:
            self.data[key] = {}
            result = self.run_synth(split, id, sketch)
            self.data[key][sketch] = result
            return result

    def merge(self, src_cache):
        for key in src_cache.data:
            
            if key not in self.data:
                self.data[key] = {}

            src_sketches = src_cache.data[key]

            for var in src_sketches:
                if var not in self.data[key]:
                    self.data[key][var] = src_sketches[var]

    def soft_query(self, split, id, sketch):

        key = split + str(id)
        if key in self.data:
            vars = self.data[key]
            if sketch in vars:
                # get result
                return vars[sketch]
            else:
                return None
        else:
            self.data[key] = {}
            return None
    
    def soft_write(self, split, id, sketch, result):
        key = split + str(id)
        self.data[key][sketch] = result


class TimedCache(SynthCache):

    def __init__(self, cache_id, dataset):
        self.dataset = dataset
        if dataset.startswith("TurkSketch"):
            self.cache_id = "TMTurk-" + cache_id
            self.cache_file = get_cache_file(self.cache_id)
            self.timeout = 2
            self.mode = "1"
        if dataset.startswith("KB13Sketch"):
            self.cache_id = "TMKB-" + cache_id
            self.cache_file = get_cache_file(self.cache_id)
            self.timeout = 5
            self.mode = "2"

        self.data = {}
        self.load()

    def run_synth(self, split, id, sketch):
        t_start = time.monotonic()
        # java -Djava.library.path=external/lib -cp external/resnax.jar:external/lib/* -ea resnax.Main 1 $1 $2 $3
        cmd = ["java", "-Djava.library.path=external/lib", "-cp", "external/resnax.jar:external/lib/*", "-ea", "resnax.Main", self.mode, split, str(id), sketch]
        try:
            out = str(subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=self.timeout))
            if "true" in out:
                result = "true"
            elif "false" in out:
                result = "false"
            elif "wrong" in out:
                result = "wrong"
            elif "null" in out:
                result = "null"
            elif "empty" in out:
                result = "empty"
        except subprocess.TimeoutExpired:
            result = "timeout"
        except subprocess.CalledProcessError:
            result = "wrong"
        except ValueError:
            result = "wrong"
            print("Value Error", split, id, sketch)
            print("Value Error!!!!!!", split, id, sketch, file=sys.stderr)

        return result, (time.monotonic() - t_start)

    def timed_query(self, split, id, sketch):
        return super().query(split, id, sketch)

    def query(self, split, id, sketch):
        return super().query(split, id, sketch)[0]

    def timed_soft_query(self, split, id, sketch):
        return super().soft_query(split, id, sketch)

    def soft_query(self, split, id, sketch):
        return super().soft_query(split, id, sketch)[0]

    def soft_write(self, split, id, sketch, result):
        assert(len(result) == 2)
        key = split + str(id)
        self.data[key][sketch] = result
