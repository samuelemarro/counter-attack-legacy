import multiprocessing
import queue
import threading
import time

import numpy as np

def _parallel_thread_function(input_queue, output_queue, thread_worker, pooler_interface):
    pooler_interface.register()

    while True:
        input_id, _input = input_queue.get()
        output = thread_worker(_input, pooler_interface)
        output_queue.put((input_id, output))

class ParallelPooler:
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
    def __init__(self, batch_worker, thread_worker, thread_count):
        self._batch_worker = batch_worker
        self._thread_count = thread_count
        self._manager = multiprocessing.Manager()
        self._pooler_interface = PoolerInterface(self._manager, thread_count)
        self._input_queue = self._manager.Queue()
        self._output_queue = self._manager.Queue()
        self._threads = []

        import dill
        #print(dill.detect.badobjects(thread_worker, depth=2))
        #dill.detect.badobjects(self._pooler_interface, depth=2)
        print(dill.detect.errors(thread_worker))

        for _ in range(thread_count):
            thread = multiprocessing.Process(target=_parallel_thread_function,
                                    args=(self._input_queue, self._output_queue, thread_worker, self._pooler_interface))

            self._threads.append(thread)
            thread.start()

        # Il pooler handler può gestire tutto, incluso il sistema di queue. Volendo si può passare di funzione in funzione
        # il pooler handler e lasciargli fare tutto
        # La funzione chiama l'handler e gli passa gli input, lui li organizza in maniera queue con i ThreadWorker (ora tutti semplici input-output)
        # Usando qualche sistema di id (forse c'è già?), lui sa automaticamente ricostruire l'output.
        # In tutto questo i thread rimangono costantemente attivi, anche se forse si potrebbe mettere un evento start/stop
        # Il BatchPooler potrebbe ricevere in __call__ il batch worker? Nah non è il caso

    def _collect_requests(self):
        requests = []

        try:
            while True:
                requests.append(self._pooler_interface.requests.get(timeout=1e-5))
        except multiprocessing.queues.Empty:
            # No more requests
            pass

        return requests

    def _put_responses(self, requests, response_outputs):
        # Match all responses with their requests
        for request in requests:
            matching_response_outputs = []
            for thread_id, output_data in response_outputs:
                # print('Thread id in output: {}'.format(thread_id))
                if request.thread_id == thread_id:
                    # print('Adding for thread {}'.format(thread_id))
                    matching_response_outputs.append(output_data)

            # print('Returning {} element(s).'.format(len(matching_response_outputs)))
            if request.batch:
                assert len(matching_response_outputs) > 0
                self._pooler_interface.response_queues[request.thread_id].put(matching_response_outputs)
            else:
                assert len(matching_response_outputs) == 1
                self._pooler_interface.response_queues[request.thread_id].put(matching_response_outputs[0])

    def _perform_middleman(self, input_count):
        collected_outputs = []

        while len(collected_outputs) < input_count:
            requests = self._collect_requests()

            print('Requests: {}. Outputs: {}/{}'.format(len(requests), len(collected_outputs), input_count))

            if requests:
                thread_ids = [request.thread_id for request in requests]

                # Check that all threads are unique
                assert len(np.unique(thread_ids)) == len(thread_ids)

                # request_inputs is a list of (thread_id, input_data) tuples
                request_inputs = []
                for request in requests:
                    request_inputs += request.get_inputs()

                input_datas = [input_data for _, input_data in request_inputs]
                # print(len(input_datas))

                output_datas = self._batch_worker(input_datas)

                assert len(input_datas) == len(output_datas)

                # Match the output data with its thread_id
                # response_outputs is a list of (thread_id, output_data) tuples
                response_outputs = []
                for (thread_id, _), output_data in zip(request_inputs, output_datas):
                    response_outputs.append((thread_id, output_data))

                assert len(request_inputs) == len(response_outputs)

                self._put_responses(requests, response_outputs)
                
            # Collect outputs
            while True:
                try:
                    collected_output = self._output_queue.get(timeout=1e-2)
                    collected_outputs.append(collected_output)
                except multiprocessing.queues.Empty:
                    break

        return collected_outputs

    def run(self, inputs):
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

        # Wait for all threads to register
        while self._pooler_interface.registration_counter.value < self._thread_count:
            time.sleep(0.01)

        for input_id, _input in enumerate(inputs):
            self._input_queue.put((input_id, _input))

        collected_outputs = self._perform_middleman(len(inputs))

        # Sort outputs
        outputs = [None] * len(inputs)

        for collected_output in collected_outputs:
            input_id, output_data = collected_output
            outputs[input_id] = output_data

        # Check that we've received all outputs
        for output in outputs:
            assert output is not None

        return outputs


    def stop(self):
        for thread in self._threads:
            thread.shutdown()

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
    def __init__(self, manager, thread_count):
        # thread_id is thread-dependent
        self.thread_id = None
        
        # All these objects are shared and thread-safe
        self.manager = manager
        self.requests = manager.Queue()
        self.response_queues = [manager.Queue() for _ in range(thread_count)]
        self.registration_lock = manager.Lock()
        self.registration_counter = manager.Value('I', 0)

    def register(self):
        with self.registration_lock:
            thread_id = self.registration_counter.value
            self.registration_counter.value = self.registration_counter.value + 1

            self.thread_id = thread_id

    def call(self, input_data, batch):
        #print('Calling from {}'.format(self.thread_id))
        
        assert self.thread_id is not None

        request = PoolerRequest(self.thread_id, input_data, batch)
        self.requests.put(request)

        return self.response_queues[self.thread_id].get()


class BatchWorker:
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


class ThreadWorker:
    def __init__(self):
        pass

    def __call__(self, _input, pooler_interface):
        pass
