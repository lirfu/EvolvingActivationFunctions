package hr.fer.zemris.utils;

import com.sun.istack.internal.NotNull;
import hr.fer.zemris.data.primitives.DataPair;
import hr.fer.zemris.genetics.Genotype;

import java.util.Random;

public class Util {
        public static String formatMiliseconds(long miliseconds) {
        if (miliseconds >= 1000L) {
            long seconds = miliseconds / 1000L;
            miliseconds -= seconds * 1000L;

            if (seconds >= 60L) {
                long minutes = seconds / 60L;
                seconds -= minutes * 60L;

                if (minutes >= 60L) {
                    long hours = minutes / 60L;
                    minutes -= hours * 60L;

                    if (hours >= 24L) {
                        long days = hours / 24L;
                        hours -= days * 24L;

                        return days + "d " + hours + "h " + minutes + "m " + seconds + "." + miliseconds + "s";
                    }
                    return hours + "h " + minutes + "m " + seconds + "." + miliseconds + "s";
                }
                return minutes + "m " + seconds + "." + miliseconds + "s";
            }
            return seconds + "." + miliseconds + "s";
        }
        return "0." + (miliseconds / 100) + (miliseconds % 100 / 10) + (miliseconds % 100 % 10) + "s";
    }

    public static String diff(String s1, String s2) {
        StringBuilder sb = new StringBuilder();
        boolean t = false;
        for (int i = 0; i < Math.min(s1.length(), s2.length()); i++) {
            char c1 = s1.charAt(i), c2 = s2.charAt(i);
            if (!t && c1 != c2) {
                sb.append('<');
                t = true;
            } else if (t && c1 == c2) {
                sb.append('>');
                t = false;
            }
            sb.append(c1);
        }
        if (s1.length() != s2.length()) {
            sb.append("xxx...");
        }
        return sb.toString();
    }
}
