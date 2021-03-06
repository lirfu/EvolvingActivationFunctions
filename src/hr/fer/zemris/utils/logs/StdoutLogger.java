package hr.fer.zemris.utils.logs;

public class StdoutLogger implements ILogger {
    private LogLevel level_;

    public StdoutLogger() {
        level_ = new LogLevel(LogLevel.INFO | LogLevel.WARNING | LogLevel.ERROR);
    }

    public StdoutLogger(LogLevel level) {
        if (level == null) {
            level = new LogLevel(LogLevel.INFO | LogLevel.WARNING | LogLevel.ERROR);
        }
        level_ = level;
    }

    private String thread_info() {
        return '[' + Thread.currentThread().getName() + "] ";
    }

    @Override
    public void d(String s) {
        if (level_.debug()) System.out.println(thread_info() + s);
    }

    @Override
    public void i(String s) {
        if (level_.info()) System.out.println(thread_info() + s);
    }

    @Override
    public void w(String s) {
        if (level_.warning()) System.out.println(thread_info() + s);
    }

    @Override
    public void e(String s) {
        if (level_.error()) System.err.println(thread_info() + s);
    }

    @Override
    public void o(Object o) {
        if (level_.info()) System.out.println(thread_info() + o.toString());
    }
}
