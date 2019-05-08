package hr.fer.zemris.utils.logs;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class SlackLoggerTest {
    @Test
    public void sendTestMessage() throws IOException {
        String url = "slack_webhook.txt";
        SlackLogger log = new SlackLogger("test_logger", url, ":ant:");

        log.d("This is a test message! :ant:");
    }
}