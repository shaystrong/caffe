# Original Author: 	Anders Krogh Mortensen
# Affilication: Dept. of Agroecology, Aarhus University
# Date: 	4 Feb. 2016
# Modified: Shay Strong 12 July 2016

import caffe
import lmdb
import numpy as np
import sys
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array

#label_lmdb = './lmdb/train/color-lmdb/'
for a in range(1,len(sys.argv)):
    label_lmdb = sys.argv[a]

    lmdb_env = lmdb.open(label_lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    heights = [];
    widths = [];
    channels = [];
    labels = [];
    keys = [];

    sanityChecks = [];

    dataMaxs = [];
    dataMins = [];
    dataMeans = [];

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)

        C, H, W = data.shape 

        dataMaxs.append(np.max(data,(1,2)))
        dataMins.append(np.min(data,(1,2)))
        dataMeans.append(np.mean(data,(1,2)))

	if ((datum.height * datum.width * datum.channels) == np.prod(data.shape)):
            sanityChecks.append(1)
        else:
            sanityChecks.append(0)

        heights.append(datum.height)
        widths.append(datum.width)
        channels.append(datum.channels)
        labels.append(datum.label)
        keys.append(key)


    maxArray = np.asarray(dataMaxs);
    minArray = np.asarray(dataMins);
    meanArray = np.asarray(dataMeans);
    print(' ');
    print('Summary of ' + label_lmdb);
    print('Entries  : ' + str(len(sanityChecks)));
    print('Passed   : ' + str(sanityChecks.count(1)));
    print('Failed   : ' + str(sanityChecks.count(0)));
    print('         : min	max   mean');
    print('-------------------------');
    print('Height   : ' + str(np.min(heights)) + '	' + str(np.max(heights)));
    print('Width    : ' + str(np.min(widths)) + '	' + str(np.max(widths)));
    print('Channel  : ' + str(np.min(channels)) + '	' + str(np.max(channels)));
    print('Label    : ' + str(np.min(labels)) + '	' + str(np.max(labels)));
    print('Data     : ');
    for i in range(np.max(channels)):
        print('  ch. ' + str(i) + '  : ' + str(np.min(minArray[:,i])) + '	' + str(np.max(maxArray[:,i])) + '	' + str(np.mean(meanArray[:,i])));
    print(' ');

