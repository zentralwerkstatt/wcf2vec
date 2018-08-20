
from multiprocessing import Process, Pipe, Queue

CLOSE_SIGNAL = 421325478

def thread_parmap(func, queue, staticdata):
    recvq, sendq = queue[0], queue[1]
    # receive static data
    data = recvq.get()
    while data != CLOSE_SIGNAL:
        sendq.put(func(data, staticdata))
        data = recvq.get()
    
    #pipe.close()
    return
        
        
def parmap(func, data, staticdata, workers=8):
    ''' Map function that distributes static data to workers before starting sequential execution.'''
    
    # init pool
    #pipes = [Pipe() for i in range(workers)]
    if workers > 1:

        queues = [(Queue(),Queue()) for i in range(workers)]

        try:
            pool = [Process(target=thread_parmap, args=(func, queues[i], staticdata,)) for i in range(workers)]

            # init processes
            for i in range(workers):
                pool[i].start()

            # send data to processes
            for i in range(len(data)):
                queues[i%workers][0].put(data[i])

            # tell to close
            for i in range(workers):
                queues[i][0].put(CLOSE_SIGNAL)

            # get data from queues
            output = list()
            for i in range(len(data)):
                output.append(queues[i%workers][1].get())

            # close all explicitly
            for i in range(workers):
                pool[i].join()

        except:
            # close all explicitly
            for i in range(workers):
                pool[i].terminate()

            raise Exception('There was a problem with the parallel function.')
    
    else: # workers == 1
        output = [func(d,staticdata) for d in data]
        
    return output

