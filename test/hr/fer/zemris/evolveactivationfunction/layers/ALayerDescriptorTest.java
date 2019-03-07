package hr.fer.zemris.evolveactivationfunction.layers;

import org.junit.Test;

import static org.junit.Assert.*;

public class ALayerDescriptorTest {
    @Test
    public void testFC() {
        String inp = "fc(123)";
        FCLayerDescriptor fc = new FCLayerDescriptor();
        fc.parse(inp);
        assertTrue("FC should parse/serialize correctly: " + fc.serialize() + " != " + inp, fc.serialize().equals(inp));
    }

    @Test
    public void testConv() {
        String inp = "conv(1,2,3,4,5)";
        ConvLayerDescriptor c = new ConvLayerDescriptor();
        c.parse(inp);
        assertTrue("Conv should parse/serialize correctly: " + c.serialize() + " != " + inp, c.serialize().equals(inp));
    }
}