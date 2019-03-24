package hr.fer.zemris.utils;

import java.io.*;

public class PythonBridge {
    private String script_path;

    public PythonBridge(String script_path) {
        this.script_path = script_path;
    }

    public Session openSession() throws IOException {
        return new Session(script_path);
    }

    public static class Session {
        private BufferedReader inp;
        private BufferedWriter out;
        private Process p;

        public Session(String script_path) throws IOException {
            p = Runtime.getRuntime().exec("python " + script_path);
            inp = new BufferedReader(new InputStreamReader(p.getInputStream()));
            out = new BufferedWriter(new OutputStreamWriter(p.getOutputStream()));
        }

        public String read() throws IOException {
            return inp.readLine();
        }

        public void write(String s) throws IOException {
            out.write(s + "\n");
            out.flush();
        }

        public void close() throws IOException {
            write("quit");
            inp.close();
            out.close();
            p.destroy();
        }
    }
}
