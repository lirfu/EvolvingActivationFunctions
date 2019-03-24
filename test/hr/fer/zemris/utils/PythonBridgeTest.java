package hr.fer.zemris.utils;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class PythonBridgeTest {
    @Test
    public void testBridge() throws IOException {
        PythonBridge bridge = new PythonBridge("python/dummy_bridge_program.py");
        PythonBridge.Session s = bridge.openSession();

        // Interact with bidge.
        String response;
        s.write("aaa");
        response = s.read();
        assertTrue("Response should be non-null for: aaa", response != null);
        assertTrue("Response should be uppercase input: " + response + " != AAA", response.equals("AAA"));

        s.write("bbb");
        response = s.read();
        assertTrue("Second response should be non-null for: bbb", response != null);
        assertTrue("Second response should be uppercase input: " + response + " != BBB", response.equals("BBB"));

        // End dummy program main loop.
        s.write("end");

        // Close the bridge.
        s.close();
    }
}