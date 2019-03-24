#!/usr/env python

import sys
import time


class JavaBridge:
    def read(self):
        return sys.stdin.readline().strip()

    def write(self, s):
        sys.stdout.write(s + '\n')
        sys.stdout.flush()

    def end(self):
        # Notify parent.
        self.write('done')
        # Wait for parent to respond.
        s = self.read()
        while s is not 'quit':
            s = self.read()
            time.sleep(.300)
