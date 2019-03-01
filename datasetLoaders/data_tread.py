import threading
import queue
import glob
import random
import time
import pickle
import zlib
import pdb 
import helper
import numpy as np 
import scipy
import scipy.misc

# def densify(data):
#     dense = np.zeros(data[0])
#     dense[data[1]] = 1
#     return dense

def list_files(pattern, filename_queue):
    filelist = glob.glob(pattern)
    while True:
        random.shuffle(filelist)
        for filename in filelist:
            filename_queue.put(filename)

def load_file(filename_queue, data_queue):
    while True:
        filename = filename_queue.get()
        # rgb_scaled = helper.read_cropped_celebA(filename)
        rgb = scipy.misc.imread(filename)
        data_queue.put((rgb, filename))
        
        # with open(filename, 'rb') as f:
        #     compressed = f.read()
        # pickled = zlib.decompress(compressed)
        # sparse = pickle.loads(pickled)
        
        # # data = sparse
        # data = {}
        # for k in sparse.keys():
        #     if str(k).split('.')[2] == 'cont': v = sparse[k][1]
        #     else: v = densify(sparse[k])
        #     data[k] = v
        # data_queue.put((rgb_scaled, filename))

counter = 0
counter_lock = threading.Lock()
def consume_data(data_queue):
    global counter
    while True:
        data_queue.get()
        with counter_lock:
            counter += 1

def returnQueue(pattern, queue_depth = 4, worker_count = 2):
    filename_queue = queue.Queue(queue_depth)
    data_queue = queue.Queue(queue_depth)
    threading.Thread(target=list_files, args=(pattern, filename_queue,), daemon=True).start()
    for i in range(worker_count):
        threading.Thread(target=load_file, args=(filename_queue, data_queue,), daemon=True).start()
    return data_queue












# def main():
#     QUEUE_DEPTH = 4
#     WORKER_COUNT = 4
#     load_file_dir = '/Users/mevlana/protobuf/datasets_3/'
#     pattern = load_file_dir+'ADS_*.npz'

#     filename_queue = queue.Queue(QUEUE_DEPTH)
#     data_queue = queue.Queue(QUEUE_DEPTH)
#     threading.Thread(target=list_files, args=(pattern, filename_queue,), daemon=True).start()
#     for i in range(WORKER_COUNT):
#         threading.Thread(target=load_file, args=(filename_queue, data_queue,), daemon=True).start()
#     # threading.Thread(target=consume_data, args=(data_queue,), daemon=True).start()
#     # pdb.set_trace()

#     # item = data_queue.get()
#     # first item takes awhile to load, but times should be stable after that
#     print('waiting for first item')
#     item = data_queue.get()
#     print('got first item')

#     start = time.time()
#     while True:
#         with counter_lock:
#             count = counter
#         print((time.time() - start) / count)
#         time.sleep(1)


# main()