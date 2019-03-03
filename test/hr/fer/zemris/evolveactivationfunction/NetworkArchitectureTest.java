package hr.fer.zemris.evolveactivationfunction;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class NetworkArchitectureTest {
    @Test
    public void parseTestFC() {
        String s = "fc(1)-fc(2)-fc(3)";
        assertTrue("String should satisfy regex.", s.matches(NetworkArchitecture.REGEX));

        NetworkArchitecture na = new NetworkArchitecture();

        assertTrue("Parse should return true.", na.parse(s));
        assertTrue("Serialization-deserialization should give same string: " + na.serialize(), s.equals(na.serialize()));
        assertTrue("Parsed wrong number of layers: " + na.getLayers().size(), na.getLayers().size() == 3);
    }

    @Test
    public void parseTestCONV() {
        String s = "conv(1)-conv(2)-conv(3)";
        assertTrue("String should satisfy regex.", s.matches(NetworkArchitecture.REGEX));

        NetworkArchitecture na = new NetworkArchitecture();

        assertTrue("Parse should return true.", na.parse(s));
        assertTrue("Serialization-deserialization should give same string: " + na.serialize(), s.equals(na.serialize()));
        assertTrue("Parsed wrong number of layers: " + na.getLayers().size(), na.getLayers().size() == 3);
    }
}