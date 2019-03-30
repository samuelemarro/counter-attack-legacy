import queue
import threading


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

    def __init__(self, batch_worker):
        self.inputs = {}
        self.outputs = {}
        self.registered_ids = []
        self.deregistered_ids = queue.Queue()
        self.batch_worker = batch_worker
        self.running = False

    def register(self):
        # Only register before execution
        assert not self.running
        new_id = threading.get_ident()
        self.registered_ids.append(new_id)
        self.inputs[new_id] = queue.Queue()
        self.outputs[new_id] = queue.Queue()

    def deregister(self):
        # Schedule for cleanup
        self.deregistered_ids.put(threading.get_ident())

    def call(self, input):
        thread_id = threading.get_ident()
        self.inputs[thread_id].put(input)
        return self.outputs[thread_id].get()

    def _get_deregistered_id(self):
        try:
            deregistered_id = self.deregistered_ids.get(timeout=1e-5)
        except queue.Empty:
            deregistered_id = None

        return deregistered_id

    def run(self):
        self.running = True
        while self.registered_ids:
            inputs = []
            active_ids = []

            for registered_id in self.registered_ids:
                try:
                     input = self.inputs[registered_id].get(timeout=1e-5)
                except queue.Empty:
                    input = None

                if input is not None:
                    inputs.append(input)
                    active_ids.append(registered_id)

            if active_ids:
                outputs = self.batch_worker(inputs)
                for active_id, output in zip(active_ids, outputs):
                    self.outputs[active_id].put(output)

            # Cleanup
            deregistered_id = self._get_deregistered_id()
            while deregistered_id is not None:
                index = self.registered_ids.index(deregistered_id)
                del self.registered_ids[index]
                del self.inputs[deregistered_id]
                del self.outputs[deregistered_id]

                deregistered_id = self._get_deregistered_id()


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


def _parallel_thread_function(pooler, thread_worker, output_queue):
    pooler.register()
    thread_worker(pooler, output_queue)
    pooler.deregister()


def run_queue_threads(batch_worker, thread_workers, input_queue, data):
    """
    Manages threading and queuing for batch and thread workers.
    """
    pooler = BatchPooler(batch_worker)
    threads = []

    output_queue = queue.Queue()

    for i, data_entry in enumerate(data):
        input_queue.put((i, data_entry))

    for thread_worker in thread_workers:
        thread = threading.Thread(target=_parallel_thread_function,
                                  args=(pooler, thread_worker, output_queue))

        threads.append(thread)
        thread.start()

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
