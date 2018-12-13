package hr.fer.zemris.utils.threading;

import java.util.LinkedList;

public class Worker {
    private String mID;

    private boolean isAlive = true;
    private final Thread mThread;
    private final LinkedList<Work> mWorkQueue;

    /**
     * Creates a new worker and starts the internal thread.
     */
    public Worker(String id) {
        mID = id;
        mWorkQueue = new LinkedList<>();

        mThread = new Thread(() -> {
            while (isAlive) {
                doWork();
                notifyAll();
            }
        });
        mThread.setDaemon(true);
        mThread.start();
    }

    private synchronized void doWork() {
        while (mWorkQueue.size() > 0) {
            mWorkQueue.pop().work();
        }

        // wait for changes in queue or kill signal
        try {
            wait(10_000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }


    /**
     * Enqueues work to workers' queue. If worker is dead, work is dropped.
     */
    public synchronized void enqueueWork(Work work) {
        if (!isAlive) return;
        mWorkQueue.push(work);
        notify();
    }

    /**
     * Returns the current queue size.
     */
    public int getWorkCount() {
        return mWorkQueue.size();
    }

    /**
     * Returns workers' name.
     */
    public String getID() {
        return mID;
    }

    /**
     * Kills the worker and notifies his listeners. May he rest in peace for all of eternity.
     */
    public synchronized void kill() {
        isAlive = false;
        notify();
    }
}
