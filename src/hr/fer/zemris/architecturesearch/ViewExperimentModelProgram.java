package hr.fer.zemris.architecturesearch;

import hr.fer.zemris.evolveactivationfunction.nn.TrainProcedure;
import hr.fer.zemris.utils.commandline.ACommand;
import hr.fer.zemris.utils.commandline.CommandLine;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;

import java.io.File;

public class ViewExperimentModelProgram {
    public static void main(String[] args) {
        if (args.length == 1) {
            TrainProcedure.displayTrainStats(new FileStatsStorage(new File(args[0])));
        } else {
            CommandLine cmd = new CommandLine();
            cmd.addCommand(new ACommand("display", "Runs the dl4j server for displaying model results in the given file. Usage: displayRuns <file-path>") {
                @Override
                public void execute(String parameters) {
                    File f = new File(parameters);
                    if (!f.exists()) {
                        System.out.println("File not found: " + parameters);
                        return;
                    }
                    TrainProcedure.displayTrainStats(new FileStatsStorage(f));
                }
            });
            cmd.addCommand(new ACommand("quit", "Stops the server and exits the command line.") {
                @Override
                public void execute(String parameters) {
                    UIServer.getInstance().stop();
                    cmd.stop();
                }
            });
            cmd.addCommand(cmd.constructHelpCommand());
            cmd.runCommand("help");
            cmd.run();
        }
    }
}
