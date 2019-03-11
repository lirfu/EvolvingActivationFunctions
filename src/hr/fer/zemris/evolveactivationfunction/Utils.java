package hr.fer.zemris.evolveactivationfunction;

import hr.fer.zemris.evolveactivationfunction.tree.nodes.*;
import hr.fer.zemris.genetics.symboregression.SymbolicTree;

public class Utils {
    public static String toLatex(SymbolicTree t) {
        String token = "<->";
        final String[] str = new String[]{token};
        t.collect(n -> {
            String s;
            switch (n.getName()) {
                case AddNode.NAME:
                    s = "(" + token + " + " + token + ")";
                    break;
                case SubNode.NAME:
                    s = "(" + token + " - " + token + ")";
                    break;
                case MulNode.NAME:
                    s = token + " \\\\cdot " + token;
                    break;
                case DivNode.NAME:
                    s = "\\\\frac{" + token + "}{" + token + "}";
                    break;
                case MinNode.NAME:
                    s = "\\\\min (" + token + "," + token + ")";
                    break;
                case MaxNode.NAME:
                    s = "\\\\max (" + token + "," + token + ")";
                    break;
                case SinNode.NAME:
                    s = "\\\\sin (" + token + ")";
                    break;
                case CosNode.NAME:
                    s = "\\\\cos (" + token + ")";
                    break;
                case TanNode.NAME:
                    s = "\\\\tan (" + token + ")";
                    break;
                case ExpNode.NAME:
                    s = "\\\\exp (" + token + ")";
                    break;
                case Pow2Node.NAME:
                    s = "(" + token + ")^2";
                    break;
                case Pow3Node.NAME:
                    s = "(" + token + ")^3";
                    break;
                case PowNode.NAME:
                    s = "(" + token + ")^{" + token + "}";
                    break;
                case LogNode.NAME:
                    s = "\\\\log (" + token + ")";
                    break;
                case InputNode.NAME:
                    s = "x";
                    break;
                case ConstNode.NAME:
                    s = String.valueOf((Double) n.getExtra());
                    break;
                case ReLUNode.NAME:
                    s = "ReLU (" + token + ")";
                    break;
                case SigmoidNode.NAME:
                    s = "\\\\sigma (" + token + ")";
                    break;
                case GaussNode.NAME:
                    s = "gauss (" + token + ")";
                    break;
                default:
                    s = "UNKNOWN";
            }
            str[0] = str[0].replaceFirst(token, s);
            return false;
        }, null);
        return str[0];
    }
}
