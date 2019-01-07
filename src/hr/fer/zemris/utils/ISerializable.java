package hr.fer.zemris.utils;

public interface ISerializable {
    /**
     * Parses given line and parses it's attribute (if present).
     */
    public boolean parse(String line);

    /**
     * Serializes object attributes onto a string.
     */
    public String serialize();
}
