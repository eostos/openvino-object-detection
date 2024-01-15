import threading

class RepetitiveInterval:
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.timer = None
        self.running = False

    def _execute(self):
        self.function()
        self.start()  # Reschedule the timer if still running

    def start(self):
        if self.running:
            # Create a timer and start it
            self.timer = threading.Timer(self.interval, self._execute)
            self.timer.start()

    def stop(self):
        self.running = False
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None