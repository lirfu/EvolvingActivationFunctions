package hr.fer.zemris.utils.logs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FileLogger implements ILogger {
    private BufferedWriter writer_;
    private boolean type_flags_;

    public FileLogger(String filepath) throws IOException {
        this(filepath, true);
    }

    public FileLogger(String filepath, boolean type_flags) throws IOException {
        new File(filepath).getParentFile().mkdirs();
        writer_ = new BufferedWriter(new FileWriter(filepath));
        type_flags_ = type_flags;
    }

    @Override
    public void d(String s) {
        try {
            if (type_flags_)
                writer_.write("[D] ");
            writer_.write(s);
            writer_.newLine();
            writer_.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void i(String s) {
        try {
            if (type_flags_)
                writer_.write("[I] ");
            writer_.write(s);
            writer_.newLine();
            writer_.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void w(String s) {
        try {
            if (type_flags_)
                writer_.write("[W] ");
            writer_.write(s);
            writer_.newLine();
            writer_.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void e(String s) {
        try {
            if (type_flags_)
                writer_.write("[E] ");
            writer_.write(s);
            writer_.newLine();
            writer_.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void o(Object o) {
        try {
            if (type_flags_)
                writer_.write("[O] ");
            writer_.write(o.toString());
            writer_.newLine();
            writer_.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void finalize() throws Throwable {
        writer_.close();
        super.finalize();
    }
}
