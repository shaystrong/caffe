import lmdb
import numpy as np
import os
import sys
caffe_root = '/home/ubuntu/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

def readLMDB(lmdbPath):
    visualize = True
    lmdb_env = lmdb.open(lmdbPath)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        #label = datum.label
        data = caffe.io.datum_to_array(datum)
        meanR=data[0,:,:].mean()
        meanG=data[1,:,:].mean()
        meanB=data[2,:,:].mean()
    return data
    