package hr.fer.zemris.utils.logs;

public class StdoutLogger implements ILogger {
    @Override
    public void logD(String s) {
        System.out.println(s);
    }

    @Override
    public void logW(String s) {
        System.out.println(s);
    }

    @Override
    public void logE(String s) {
        System.err.println(s);
    }

    @Override
    public void logO(Object o) {
        System.out.println(o.toString());
    }
}
