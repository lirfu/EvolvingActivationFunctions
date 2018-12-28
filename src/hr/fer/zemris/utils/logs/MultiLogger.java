package hr.fer.zemris.utils.logs;

public class MultiLogger implements ILogger {
    private ILogger[] loggers_;

    public MultiLogger(ILogger... loggers) {
        loggers_ = loggers;
    }

    @Override
    public void d(String s) {
        for (ILogger l : loggers_)
            l.d(s);
    }

    @Override
    public void i(String s) {
        for (ILogger l : loggers_)
            l.i(s);
    }

    @Override
    public void w(String s) {
        for (ILogger l : loggers_)
            l.w(s);
    }

    @Override
    public void e(String s) {
        for (ILogger l : loggers_)
            l.e(s);
    }

    @Override
    public void o(Object o) {
        for (ILogger l : loggers_)
            l.o(o);
    }
}
