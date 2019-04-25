import multiprocessing
import queue
import threading
import time

import numpy as np

class PoolerHandler:
    def __init__(self, batch_worker, thread_worker_count):
        self._thread_worker_count = thread_worker_count
        self._manager = multiprocessing.Manager()
        self._pooler_interface = PoolerInterface(self._manager, thread_worker_count)
        self._batch_pooler = BatchPooler(batch_worker, self._manager, self._pooler_interface, self._thread_worker_count)
        self._input_queue = self._manager.Queue()
        self._output_queue = self._manager.Queue()
        self._threads = []

        # Inizializza i thread
        # Il pooler handler può gestire tutto, incluso il sistema di queue. Volendo si può passare di funzione in funzione
        # il pooler handler e lasciargli fare tutto
        # La funzione chiama l'handler e gli passa gli input, lui li organizza in maniera queue con i ThreadWorker (ora tutti input queue-output queue)
        # Usando qualche sistema di id (forse c'è già?), lui sa automaticamente ricostruire l'output.
        # In tutto questo i thread rimangono costantemente attivi, anche se forse si potrebbe mettere un evento start/stop
        # Il BatchPooler potrebbe ricevere in __call__ il batch worker? Nah non è il caso

    def run_threads(self, inputs):

        # Ensure that the input and output queues are empty
        try:
            self._input_queue.get(timeout=1e-2)
            assert False
        except multiprocessing.queues.Empty:
            pass

        try:
            self._output_queue.get(timeout=1e-2)
            assert False
        except multiprocessing.queues.Empty:
            pass

        for input_id, _input in enumerate(inputs):
            self._input_queue.put((input_id, _input))

class PoolerRequest:
    def __init__(self, thread_id, input_data, batch):
        self.thread_id = thread_id
        self.input_data = input_data
        self.batch = batch

    def get_inputs(self):
        if self.batch:
            inputs = []
            for element in self.input_data:
                inputs.append((self.thread_id, element))
        else:
            inputs = [(self.thread_id, self.input_data)]

        return inputs

class PoolerInterface:
    """
    A 100% pickable interface for communicating with BatchPooler
    (which isn't pickable because it contains a BatchWorker).
    """
    def __init__(self, manager, thread_worker_count):
        # thread_id is thread-dependent
        self.thread_id = None
        
        # All these objects are shared and thread-safe
        self.manager = manager
        self.requests = manager.Queue()
        self.output_queues = [manager.Queue() for _ in range(thread_worker_count)]
        self.registration_lock = manager.Lock()
        self.registration_counter = manager.Value('I', 0)
        self.deregistration_counter = manager.Value('I', 0)

    def register(self):
        with self.registration_lock:
            thread_id = self.registration_counter.value
            self.registration_counter.value = self.registration_counter.value + 1

            self.thread_id = thread_id

    def deregister(self):
        # print('Deregistering')
        with self.registration_lock:
            self.deregistration_counter.value = self.deregistration_counter.value + 1

    def call(self, input_data, batch):
        #print('Calling from {}'.format(self.thread_id))
        
        assert self.thread_id is not None

        request = PoolerRequest(self.thread_id, input_data, batch)
        self.requests.put(request)

        return self.output_queues[self.thread_id].get()


class BatchPooler:
    """
    Manages workers by acting as a middleman between thread workers
    (which provide inputs) and the batch worker (which provides outputs).
    When running, it collects several inputs from the thread workers and
    uses the batch worker to compute the outputs. Once the outputs are ready,
    it returns all the outputs to the corresponding thread workers. This is
    especially useful when you have some methods that are highly parallelizable
    (batch workers), but need to be used by several non-parallel methods
    (thread workers).
    """

    def __init__(self, batch_worker, manager, pooler_interface, thread_worker_count):
        self.batch_worker = batch_worker
        self.pooler_interface = pooler_interface
        self.thread_worker_count = thread_worker_count

    def _collect_requests(self):
        requests = []

        try:
            while True:
                requests.append(self.pooler_interface.requests.get(timeout=1e-5))
        except multiprocessing.queues.Empty:
            # No more requests
            pass

        return requests


    def run(self):
        # Wait for all threads to register
        while self.pooler_interface.registration_counter.value < self.thread_worker_count:
            time.sleep(0.01)

        # print('All threads registered')

        while self.pooler_interface.deregistration_counter.value < self.thread_worker_count:
            requests = self._collect_requests()

            # print('Requests: {}. Deregistered: {}/{}'.format(len(requests), self.pooler_interface.deregistration_counter.value, self.pooler_interface.registration_counter.value))

            if requests:
                thread_ids = [request.thread_id for request in requests]

                # Check that all threads are unique
                assert len(np.unique(thread_ids)) == len(thread_ids)

                # inputs is a list of (thread_id, input_data) tuples
                inputs = []
                for request in requests:
                    inputs += request.get_inputs()

                input_datas = [input_data for _, input_data in inputs]
                # print(len(input_datas))

                output_datas = self.batch_worker(input_datas)

                assert len(input_datas) == len(output_datas)

                # Match the output data with its thread_id
                # outputs is a list of (thread_id, output_data) tuples
                outputs = []
                for (thread_id, _), output_data in zip(inputs, output_datas):
                    outputs.append((thread_id, output_data))

                assert len(inputs) == len(outputs)

                # Match all outputs with their requests
                for request in requests:
                    matching_outputs = []
                    for thread_id, output_data in outputs:
                        # print('Thread id in output: {}'.format(thread_id))
                        if request.thread_id == thread_id:
                            # print('Adding for thread {}'.format(thread_id))
                            matching_outputs.append(output_data)

                    
                    # print('Returning {} element(s).'.format(len(matching_outputs)))
                    if request.batch:
                        assert len(matching_outputs) > 0
                        self.pooler_interface.output_queues[request.thread_id].put(matching_outputs)
                    else:
                        assert len(matching_outputs) == 1
                        self.pooler_interface.output_queues[request.thread_id].put(matching_outputs[0])

        # print('No more registered ids')

class BatchWorker:
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


class ThreadWorker:
    def __init__(self):
        pass

    def __call__(self, pooler, return_queue):
        pass


def _parallel_thread_function(pooler_interface, thread_worker, output_queue):
    pooler_interface.register()
    thread_worker(pooler_interface, output_queue)
    pooler_interface.deregister()


def run_queue_threads(manager, batch_worker, thread_workers, input_queue, data):
    """
    Manages threading and queuing for batch and thread workers.
    """
    pooler = BatchPooler(batch_worker, manager, len(thread_workers))
    pooler_interface = pooler.pooler_interface
    threads = []

    output_queue = manager.Queue()

    for i, data_entry in enumerate(data):
        input_queue.put((i, data_entry))

    for thread_worker in thread_workers:
        import dill.detect as detect
        thread = multiprocessing.Process(target=_parallel_thread_function,
                                  args=(pooler_interface, thread_worker, output_queue))

        threads.append(thread)
        thread.start()

    # print('About to run!')
    pooler.run()
    for thread in threads:
        thread.join()

    outputs = []

    while True:
        try:
            outputs.append(output_queue.get(timeout=1e-5))
        except queue.Empty:
            break

    # Sort the outputs
    outputs = sorted(outputs, key=lambda output: output[0])

    return [output[1] for output in outputs]
