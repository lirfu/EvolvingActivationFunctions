package hr.fer.zemris.utils.logs;

import java.io.File;
import java.io.IOException;

public class FileLoggerTest {
    private static final String file = System.getProperty("user.dir") + File.separator + "test" + File.separator + "gen" + File.separator + "test.txt";
    private static FileLogger log;

    @org.junit.BeforeClass
    public static void setUp() throws IOException {
        System.out.println(file);
        File f = new File(file);
        f.getParentFile().mkdirs();
        log = new FileLogger(file);
    }


    @org.junit.Test
    public void logD() throws Exception {
        log.d("This line is totally ok!\nEven the new line works.");
    }

    @org.junit.Test
    public void logO() throws Exception {
        Object o = new Object() {
            @Override
            public String toString() {
                return "@Obj1 [This][is a][object]";
            }
        };
        log.o(o);
    }

}