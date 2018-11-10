package hr.fer.zemris.utils.commandline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.TreeMap;

public class CommandLine {
    private boolean mRunning = false;
    private TreeMap<String, ACommand> mCommands = new TreeMap<>();

    public void addCommand(ACommand command) {
        if (mCommands.containsKey(command.getName()))
            throw new IllegalStateException("Command " + command.getName() + " is already defined!");
        mCommands.put(command.getName(), command);
    }

    public ACommand constructHelpCommand() {
        return new ACommand("help", "Displays this help message.") {
            @Override
            public void execute(String parameters) {
                System.out.println("Available commands:");

                int longestCmdName = 0;
                for (String cmdName : mCommands.keySet())
                    if (cmdName.length() > longestCmdName) longestCmdName = cmdName.length();

                for (ACommand command : mCommands.values()) {
                    StringBuilder output = new StringBuilder();
                    output.append(command.getName());
                    for (int i = 0; i < longestCmdName - command.getName().length() + 1; i++)
                        output.append(' ');
                    output.append(command.getDescription());
                    System.out.println(output.toString());
                }
            }
        };
    }

    public boolean ismRunning() {
        return mRunning;
    }

    public void stop() {
        mRunning = false;
    }

    public void run() {
        run(false);
    }

    public void run(boolean silentPrompt) {
        mRunning = true;

        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

            while (mRunning) {
                if (!silentPrompt)
                    System.out.print("> ");

                String command = null;
                StringBuilder params = new StringBuilder();
                String input = reader.readLine();

                if (input == null) // Input stream died so you can end.
                    break;

                for (char c : input.toCharArray()) {
                    if (c == ' ' && command == null) { // Command name is read, rest goes to parameters.
                        command = params.toString();
                        params = new StringBuilder();
                    }
                    params.append(c);
                }
                if (command == null) { // No params were inputted (no spaces)
                    command = params.toString();
                    params = new StringBuilder();
                }

                ACommand cmd = mCommands.get(command);
                if (cmd == null)
                    System.out.println("Unknown command: " + command);
                else
                    cmd.execute(params.toString().trim());
            }

            reader.close();
        } catch (IOException e) {
            System.out.println("Command line stream error!");
        }
    }
}
