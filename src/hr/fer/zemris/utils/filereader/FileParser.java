package hr.fer.zemris.utils.filereader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;


public class FileParser<T extends IParseable> {
    /**
     * Reads the file at given filepath and parses lines as objects of given template.
     * Calls parseLine on template for each line in file and adds parsed object to list.
     * Ignores null (enables implementation of comments).
     * @param filepath
     * @param template
     * @return
     * @throws IOException
     */
    public LinkedList<T> parse(String filepath, T template) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        LinkedList<T> data = new LinkedList<>();

        String input;
        while ((input = reader.readLine()) != null) {
            T d = (T) template.parseLine(input);
            if (d != null)
                data.push(d);
        }

        reader.close();
        return data;
    }
}
