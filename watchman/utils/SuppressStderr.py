import os
import sys


class SuppressStderr:
    def __init__(self, to_file=None):
        self.stderr = sys.stderr.fileno()
        self.stderr_save = os.dup(self.stderr)
        self.to_file = to_file or os.devnull

    def __enter__(self):
        self.stderr_log = open(self.to_file, 'w')
        os.dup2(self.stderr_log.fileno(), self.stderr)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.stderr_save, self.stderr)
        self.stderr_log.close()
