package hr.fer.zemris.utils;


import hr.fer.zemris.genetics.symboregression.IInstantiable;
import com.sun.istack.NotNull;
import com.sun.istack.Nullable;

import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Pattern;

public class Utilities {
    public static final String KEY_VALUE_SIMPLE_REGEX = "[\t :]+";
    public static final Pattern KEY_VALUE_REGEX = Pattern.compile("([^\t :]+)[\t :]+([^#]+)#*.*");
    public static final Pattern ARRAY_REGEX = Pattern.compile("\\{ *(.+) *\\}");
    public static final String ARRAY_SEPARATOR = " *, *";

    public static void permuteArray(@NotNull Object[] array, int n, @Nullable Random rand) {
        if (rand == null) rand = new Random();
        for (int i = 0; i < n; i++) {
            int x = rand.nextInt(array.length);
            int y = rand.nextInt(array.length);
            Object t = array[x];
            array[x] = array[y];
            array[y] = t;
        }
    }

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

    public static float[] initEmptyArray(int size, float value) {
        float[] arr = new float[size];
        for (int i = 0; i < size; i++)
            arr[i] = value;
        return arr;
    }

    public static double[] initEmptyArray(int size, double value) {
        double[] arr = new double[size];
        for (int i = 0; i < size; i++)
            arr[i] = value;
        return arr;
    }

    public static <T> T[] initEmptyArray(int size, T value) {
        T[] arr = (T[]) new Object[size];
        for (int i = 0; i < size; i++)
            arr[i] = value;
        return arr;
    }

    public static <T> LinkedList<T> listByRepeating(T element, int times) {
        LinkedList<T> list = new LinkedList<>();
        for (int i = 0; i < times; i++) {
            list.add(element);
        }
        return list;
    }

    public static <T> ArrayList<T> arrayByRepeating(T element, int times) {
        ArrayList<T> arr = new ArrayList<>();
        for (int i = 0; i < times; i++) {
            arr.add(element);
        }
        return arr;
    }

    public static <T extends Comparable> ArrayList<Pair<T, Integer>> sortWithIndices(T[] array) {
        ArrayList<Pair<T, Integer>> result = new ArrayList<>();
        for (int i = 0; i < array.length; i++) {
            result.add(new Pair<>(array[i], i));
        }

        result.sort(Comparator.comparing(Pair::getKey));

        return result;
    }

    public static Float[] objectifyArray(float[] array) {
        Float[] result = new Float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    public static Double[] objectifyArray(double[] array) {
        Double[] result = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    public static String join(char c, int[] arr) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(arr[i]);
        }
        return sb.toString();
    }
}
