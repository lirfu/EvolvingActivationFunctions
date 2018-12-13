package hr.fer.zemris.utils.logs;

public class MultiLogger implements ILogger {
    private ILogger[] loggers_;

    public MultiLogger(ILogger... loggers) {
        loggers_ = loggers;
    }

    @Override
    public void logD(String s) {
        for (ILogger l : loggers_)
            l.logD(s);
    }

    @Override
    public void logW(String s) {
        for (ILogger l : loggers_)
            l.logW(s);
    }

    @Override
    public void logE(String s) {
        for (ILogger l : loggers_)
            l.logE(s);
    }

    @Override
    public void logO(Object o) {
        for (ILogger l : loggers_)
            l.logO(o);
    }
}
