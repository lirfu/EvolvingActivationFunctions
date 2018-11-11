package hr.fer.zemris.data;

import com.sun.istack.internal.NotNull;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Reads the specified file line-by-line.
 * When EOF is reached, <code>reset()</code> should be called to loop back to the start of the file.
 */
public class Reader extends APipe<Object, String> {
    private String filepath_;
    private BufferedReader reader_;

    /**
     * Reads the specified file line-by-line.
     *
     * @param filepath Path to the data file.
     */
    public Reader(@NotNull String filepath) throws FileNotFoundException {
        filepath_ = filepath;
        openReader();
    }

    /**
     * Open the file reader.
     */
    private void openReader() throws FileNotFoundException {
        reader_ = new BufferedReader(new FileReader(filepath_));
    }

    /**
     * Gets the next line of the file.
     * Returns <code>null</code> if end of file is reached.
     */
    @Override
    public String get() {
        try {
            return reader_.readLine();
        } catch (IOException e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    /**
     * Resets the reader to start of the file.
     */
    @Override
    public void reset() {
        try {
            reader_.close();
            openReader();
        } catch (IOException e) {
            throw new RuntimeException(e.getMessage());
        }
    }

    @Override
    public Reader clone() {
        try {
            return new Reader(filepath_);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e.getMessage());
        }
    }
}
