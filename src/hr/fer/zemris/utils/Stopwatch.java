package hr.fer.zemris.utils;

public class Stopwatch {
    private long start;
    private long lap_start;

    /**
     * Start the timer.
     * Resets internals to the current time point.
     */
    public void start() {
        start = lap_start = System.currentTimeMillis();
    }

    /**
     * Stop the timer and return total time.
     *
     * @return Total elapsed time.
     */
    public long stop() {
        return System.currentTimeMillis() - start;
    }

    /**
     * Return lap time and start new lap.
     * Doesn't affect total time.
     *
     * @return Lap time.
     */
    public long lap() {
        long now = System.currentTimeMillis();
        long diff = now - lap_start;
        lap_start = now;
        return diff;
    }
}
