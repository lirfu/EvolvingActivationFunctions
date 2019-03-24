
from java_bridge import JavaBridge

bridge = JavaBridge()

s = bridge.read()
while s is not 'end':
    bridge.write(s.upper())
    s = bridge.read()

bridge.end()
