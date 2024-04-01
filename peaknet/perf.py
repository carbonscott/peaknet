import time

class Timer:
    def __init__(self, tag = None, is_on = True):
        self.tag   = tag
        self.is_on = is_on
        self.duration = None

    def __enter__(self):
        if self.is_on:
            self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_on:
            self.end = time.monotonic()
            self.duration = self.end - self.start
            if self.tag is not None:
                print(f"{self.tag}, duration: {self.duration} sec.")
