package hr.fer.zemris.utils.logs;

public interface ILogger {
    /**
     * Log debug.
     */
    public void d(String s);


    /**
     * Log info.
     */
    public void i(String s);

    /**
     * Log warning
     */
    public void w(String s);

    /**
     * Log error
     */
    public void e(String s);

    /**
     * Log object
     */
    public void o(Object o);
}
