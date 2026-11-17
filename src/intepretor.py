# Adapted from the HORN implementation of Blum, Koudijs, Ozaki and Touileb,
# "Learning Horn envelopes via queries from language models",
# Int. J. Approx. Reason. 171 (2024) 109026, https://doi.org/10.1016/j.ijar.2023.109026
# Code base: https://github.com/ZofieB/MT_UiB (S. Blum). See README.md.
# Changes: removed sentence/template logic; maps binary vectors directly to model input tensors

import numpy as np
import csv


class Intepretor:
    def __init__(self, filePaths):
        self.lookupTable = self.formatLookupTable(filePaths)
        self.lengths = {i[0]: len(i[1][0]) for i in self.lookupTable.items()}
        self.attributes = [x for x in self.lookupTable]
        self.splitIndexes = [sum(list(self.lengths.values())[:i+1]) for i in range(len(self.lengths))]

    def formatLookupTable(self, filePaths):
        # {attribute: ([values], neutralCase)}
        lookup = {}
        for path in filePaths:
            data = []
            with open(path, mode='r', encoding='UTF-8') as f:
                csvFile = csv.reader(f)
                for lines in csvFile:
                    data.append(lines[0])
            lookup[data[0]] = (data[1:-1], data[-1])
        return lookup

    def binaryToAttribute(self, type, bin):
        attributeLookup = self.lookupTable[type]
        nonzero = np.nonzero(bin)[0]
        if len(nonzero) > 0:
            return attributeLookup[0][nonzero[0]]
        else:
            return attributeLookup[1]  # neutral/unknown value

    def binaryToModelInput(self, bin):
        """
        Decode a binary feature vector into the values needed to build the model input.

        Binary layout (matches input CSV order, labels excluded):
          bits  0-19  : HOUR one-hot (hours 5–24)
          bits 20-24  : CAPACITY one-hot (bins: 350, 600, 1000, 1500, 2400 veh/hr)

        Returns a dict with:
          'hour_onehot' : float32 array of length 20 (ready to place in x_all)
          'cap_raw'     : float, raw capacity in veh/hr (caller scales it)
        """
        bin = np.array(bin, dtype=np.float32)
        n_label_bits = list(self.lengths.values())[-1]
        feature_vec = bin[:-n_label_bits]

        hour_onehot = feature_vec[:20].astype(np.float32)

        cap_bits = feature_vec[20:25]
        BIN_MIDPOINTS = [350.0, 600.0, 1000.0, 1500.0, 2400.0]
        NEUTRAL_CAP_RAW = 1500.0
        idx = int(np.argmax(cap_bits)) if cap_bits.any() else -1
        cap_raw = BIN_MIDPOINTS[idx] if idx >= 0 else NEUTRAL_CAP_RAW

        return {'hour_onehot': hour_onehot, 'cap_raw': cap_raw}
