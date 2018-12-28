package hr.fer.zemris.utils.logs;

public class DevNullLogger implements ILogger {
    @Override
    public void d(String ignore) {
    }

    @Override
    public void i(String s) {
    }

    @Override
    public void w(String ignore) {
    }

    @Override
    public void e(String ignore) {
    }

    @Override
    public void o(Object ignore) {
    }
}
