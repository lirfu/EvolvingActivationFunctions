package hr.fer.zemris.data;

public interface ISerializable {
    /**
     * Parses given string and populates attributes.
     */
    public void parse(String s);

    /**
     * Serializes object attributes onto a string.
     */
    public String serialize();
}
