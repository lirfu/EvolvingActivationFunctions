package hr.fer.zemris.utils.threading;

import java.util.LinkedList;

public class Worker {
    private int mID;

    private boolean isAlive = true;
    private final Thread mThread;
    private final LinkedList<Work> mWorkQueue;
//    private ArrayList<Listener> mListeners;

    public Worker(int id) {
        mID = id;
        mWorkQueue = new LinkedList<>();
//        mListeners = new ArrayList<>();

        mThread = new Thread(() -> {
            while (isAlive) {
                doWork();
//                notifyListeners(mWorkQueue.size());
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

    public synchronized void enqueueWork(Work work) {
        mWorkQueue.push(work);
        notify();
    }

    public int getWorkCount() {
        return mWorkQueue.size();
    }

    public int getID() {
        return mID;
    }

    public synchronized void kill() {
        isAlive = false;
        notify();
    }

//    private void notifyListeners(int workLeft) {
//        for (Listener l : mListeners)
//            l.notify(mID, workLeft);
//    }
//
//    public void registerListener(Listener listener) {
//        mListeners.add(listener);
//    }
//
//    public void unregisterListener(Listener listener) {
//        mListeners.remove(listener);
//    }
//
//    public interface Listener {
//        public void notify(int workerID, int workLeft);
//    }
}
