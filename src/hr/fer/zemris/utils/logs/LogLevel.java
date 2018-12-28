package hr.fer.zemris.utils.logs;

public class LogLevel {
    public static final int INFO = 0x1;
    public static final int WARNING = 0x3;
    public static final int ERROR = 0x7;
    public static final int DEBUG = 0xf;

    private int flags_;

    public LogLevel(int flags) {
        flags_ = flags;
    }

    public boolean debug() {
        return (flags_ & DEBUG) > 0;
    }

    public boolean info() {
        return (flags_ & INFO) > 0;
    }

    public boolean warning() {
        return (flags_ & WARNING) > 0;
    }

    public boolean error() {
        return (flags_ & ERROR) > 0;
    }
}
