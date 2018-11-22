package hr.fer.zemris.utils.logs;

public class DevNullLogger implements ILogger {
    @Override
    public void logD(String ignore) {
    }

    @Override
    public void logW(String ignore) {
    }

    @Override
    public void logE(String ignore) {
    }

    @Override
    public void logO(Object ignore) {
    }
}
