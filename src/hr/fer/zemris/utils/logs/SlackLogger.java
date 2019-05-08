package hr.fer.zemris.utils.logs;

import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;

public class SlackLogger implements ILogger {
    private String webhook_url_;
    private String name_;
    private String icon_emoji_ = "";

    /**
     * Initialize the Slack logger.
     *
     * @param name                 Name displayed at start of every message.
     * @param webhook_url_filepath Path to file containing the Slack webhook URL.
     * @throws IOException If the file couldn't be opened or is malformed.
     */
    public SlackLogger(String name, String webhook_url_filepath) {
        name_ = name;

        // Read webhook url from file.
        // Skip empty lines and comment lines
        try {
            BufferedReader reader = new BufferedReader(new FileReader(webhook_url_filepath));
            String input;
            do {
                input = reader.readLine().trim();
            } while (input.isEmpty() || input.startsWith("#"));
            reader.close();
            webhook_url_ = input;
        } catch (FileNotFoundException e) {
            System.err.println("Webhook file " + webhook_url_filepath + " not found!");
            System.err.println("Generating the template file and throwing 'FileNotFoundException' exception.");
            try {
                BufferedWriter writer = new BufferedWriter(new FileWriter("slack_webhook.txt"));
                writer.write("# Paste your webhook URL of following form:");
                writer.newLine();
                writer.write("https://hooks.slack.com/services/SOME_CHARACTERS");
                writer.newLine();
                writer.flush();
                writer.close();
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    /**
     * Initialize the Slack logger.
     *
     * @param name                 Name displayed at start of every message.
     * @param webhook_url_filepath Path to file containing the Slack webhook URL.
     * @param icon_emoji           Emoji displayed with the message.
     * @throws Exception If the file couldn't be opened or is malformed.
     */
    public SlackLogger(String name, String webhook_url_filepath, String icon_emoji) {
        this(name, webhook_url_filepath);
        icon_emoji_ = icon_emoji;
    }

    private void sendMessage(String msg) {
        msg = '[' + name_ + "] " + msg;

        SlackMessage message = new SlackMessage(name_, msg, icon_emoji_);

        CloseableHttpClient client = HttpClients.createDefault();
        HttpPost httpPost = new HttpPost(webhook_url_);

        try {
            ObjectMapper objectMapper = new ObjectMapper();
            String json = objectMapper.writeValueAsString(message);

            StringEntity entity = new StringEntity(json);
            httpPost.setEntity(entity);
            httpPost.setHeader("Accept", "application/json");
            httpPost.setHeader("Content-type", "application/json");

            client.execute(httpPost);
            client.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void d(String s) {
        sendMessage(":ant: " + s);
    }

    @Override
    public void i(String s) {
        sendMessage(s);
    }

    @Override
    public void w(String s) {
        sendMessage(":warning: " + s);
    }

    @Override
    public void e(String s) {
        sendMessage(":fire: " + s);
    }

    @Override
    public void o(Object o) {
        sendMessage(":package: " + o.toString());
    }


    private class SlackMessage implements Serializable {
        private String username;
        private String text;
        private String icon_emoji;

        public SlackMessage(String username, String text, String icon_emoji) {
            this.username = username;
            this.text = text;
            this.icon_emoji = icon_emoji;
        }

        public String getUsername() {
            return username;
        }

        public String getText() {
            return text;
        }

        public String getIconEmoji() {
            return icon_emoji;
        }
    }
}
