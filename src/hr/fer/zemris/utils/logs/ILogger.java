package hr.fer.zemris.utils.logs;

public interface ILogger {
    /**
     * Log debug.
     */
    public void logD(String s);


    /**
     * Log warning
     */
    public void logW(String s);

    /**
     * Log error
     */
    public void logE(String s);

    /**
     * Log object
     */
    public void logO(Object o);
}
