# Common interface
class ILogger:
    def d(self, string):
        raise NotImplemented()

# Multiple destination logger
class MultiLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def d(self, string):
        for l in self.loggers:
            l.d(string)

# STDOUT logger
class StdLogger:
    def d(self, string):
        print(string)

# File logger
class FileLogger:
    def __init__(self, filepath):
        self.file = open(filepath, 'rw')

    def d(self, string):
        self.file.write(string + '\n')
