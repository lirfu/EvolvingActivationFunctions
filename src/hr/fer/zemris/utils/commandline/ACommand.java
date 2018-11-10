package hr.fer.zemris.utils.commandline;

public abstract class ACommand {
    private String mCommandName;
    private String mDescription;
    public ACommand(String name, String description){mCommandName = name;mDescription = description;}
    public String getName(){return mCommandName;}

    public String getDescription() {
        return mDescription;
    }

    public abstract void execute(String parameters);

    @Override
    public boolean equals(Object o) {
        return o instanceof ACommand && mCommandName.equals(((ACommand)o).mCommandName);
    }
}
