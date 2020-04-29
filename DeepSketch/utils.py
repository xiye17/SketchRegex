# utils.py
import time
import pickle

def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)

class TimeLogger(object):
    
    def __init__(self):
        self.time_start = time.time()
    
    def restart(self):
        self.time_start = time.time()
    
    def log(self, msg):
        print("{} cost {}s".format(msg, time.time() - self.time_start))

# Bijection between objects and integers starting at 0. Useful for mapping
# labels, features, etc. into coordinates of a vector space.
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

    def save_to_file(self, filename):
        obj = {
            'objs_to_ints': self.objs_to_ints,
            'ints_to_objs': self.ints_to_objs
        }
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    
    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        self.objs_to_ints = obj['objs_to_ints']
        self.ints_to_objs = obj['ints_to_objs']


# Map from objects to doubles that has a default value of 0 for all elements
# Relatively inefficient (dictionary-backed); useful for sparse encoding of things like gradients, but shouldn't be
# used for dense things like weight vectors (instead use an Indexer over the objects and use a numpy array to store the
# values)
class Counter(object):
    def __init__(self):
        self.counter = {}

    def __repr__(self):
        return str([str(key) + ": " + str(self.get_count(key)) for key in self.counter.keys()])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.counter)

    def keys(self):
        return self.counter.keys()

    def get_count(self, key):
        if key in self.counter:
            return self.counter[key]
        else:
            return 0

    def increment_count(self, obj, count):
        if obj in self.counter:
            self.counter[obj] = self.counter[obj] + count
        else:
            self.counter[obj] = count

    def increment_all(self, objs_list, count):
        for obj in objs_list:
            self.increment_count(obj, count)

    def set_count(self, obj, count):
        self.counter[obj] = count

    def add(self, otherCounter):
        for key in otherCounter.counter.keys():
            self.increment_count(key, otherCounter.counter[key])

    # Bad O(n) implementation right now
    def argmax(self):
        best_key = None
        for key in self.counter.keys():
            if best_key is None or self.get_count(key) > self.get_count(best_key):
                best_key = key
        return best_key


# Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
# elements after every insertion operation. Insertion is O(n) (list is maintained in
# sorted order), access is O(1). Still fast enough for practical purposes for small beams.
class Beam(object):
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

    # Adds the element to the beam with the given score if the beam has room or if the score
    # is better than the score of the worst element currently on the beam
    def add(self, elt, score):
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list contains the item with a lower score, remove it
        # i = 0
        # while i < len(self.elts):
        #     if self.elts[i] == elt and score > self.scores[i]:
        #         del self.elts[i]
        #         del self.scores[i]
        #     i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def head(self):
        return self.elts[0]


# Indexes a string feat using feature_indexer and adds it to feats.
# If add_to_indexer is true, that feature is indexed and added even if it is new
# If add_to_indexer is false, unseen features will be discarded
def maybe_add_feature(feats, feature_indexer, add_to_indexer, feat):
    if add_to_indexer:
        feats.append(feature_indexer.get_index(feat))
    else:
        feat_idx = feature_indexer.index_of(feat)
        if feat_idx != -1:
            feats.append(feat_idx)


# Computes the dot product over a list of features (i.e., a sparse feature vector)
# and a weight vector (numpy array)
def score_indexed_features(feats, weights):
    score = 0.0
    for feat in feats:
        score += weights[feat]
    return score


##################
# Tests
def test_counter():
    print("TESTING COUNTER")
    ctr = Counter()
    ctr.increment_count("a", 5)
    ctr.increment_count("b", 3)
    print(repr(ctr.get_count("a")) + " should be 5")
    ctr.increment_count("a", 5)
    print(repr(ctr.get_count("a")) + " should be 10")
    print(str(ctr.counter))
    for key in ctr.counter.keys():
        print(key)
    ctr2 = Counter()
    ctr2.increment_count("a", 3)
    ctr2.increment_count("c", 4)
    ctr.add(ctr2)
    print("%s should be ['a: 13', 'c: 4', 'b: 3']" % ctr)


def test_beam():
    print("TESTING BEAM")
    beam = Beam(3)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    print("Should contain b, c, a: %s" % beam)
    beam.add("e", 8)
    beam.add("f", 6.5)
    print("Should contain e, b, f: %s" % beam)
    beam.add("f", 9.5)
    print("Should contain f, e, b: %s" % beam)

    beam = Beam(5)
    beam.add("a", 5)
    beam.add("b", 7)
    beam.add("c", 6)
    beam.add("d", 4)
    print("Should contain b, c, a, d: %s" % beam)
    beam.add("e", 8)
    beam.add("f", 6.5)
    print("Should contain e, b, f, c, a: %s" % beam)

if __name__ == '__main__':
    test_counter()
    test_beam()
