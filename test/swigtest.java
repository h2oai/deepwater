package test.swig;

import java.util.Random;
import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

public class swigtest {
  static {
    System.loadLibrary("swigtest");
  }

  public static String FILE__() {
    return Thread.currentThread().getStackTrace()[2].getFileName();
  }

  public static int LINE__() {
    return Thread.currentThread().getStackTrace()[2].getLineNumber();
  }

  public static String LG__() {
    DateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
    return "[" + dateFormat.format(new Date()) + "] " + FILE__() + ":";
  }

  public static void main(String argv[]) {
    test.int_input_test(42);
    
    if (test.int_return_test() == -35) 
      System.out.println(LG__() + LINE__() + ": " + "int_return_test passed");
    else
      System.out.println(LG__() + LINE__() + ": " + "int_return_test failed");

    Random randnum = new Random();
    randnum.setSeed(42);
    int n = 10;
    int[] int_test = new int[n];
    for (int i = 0; i < n; i++) {
      int_test[i] = randnum.nextInt() % 1000;
    }
    test.int_arr_input_test(int_test, n);

    test.float_input_test(42.42f);
    
    if (Math.abs(test.float_return_test() - 42.42) < 1e-5) 
      System.out.println(LG__() + LINE__() + ": " + "float_return_test passed");
    else
      System.out.println(LG__() + LINE__() + ": " + "float_return_test failed");

    float[] test_arr = {0.894f, 0.223f, 0.009f, 0.343f, 0.826f,
                        0.601f, 0.201f, 0.76f, 0.65f, 0.545f};
    float[] arr = test.float_arr_return_test();
    for (int i = 0; i < 10; i++) {
      if (Math.abs(test_arr[i] - arr[i]) > 1e-5)
        System.out.println(LG__() + LINE__() + ": " + "float_arr_return_test failed");    
    }
    System.out.println(LG__() + LINE__() + ": " + "float_arr_return_test passed");

    float[] float_test = new float[n];
    for (int i = 0; i < n; i++) {
      float_test[i] = randnum.nextFloat() % 1000;
    }
    test.float_arr_input_test(float_test, n);

    test.string_input_test("string_input_test");

    if (test.string_return_test().equals("string_return_test"))
      System.out.println(LG__() + LINE__() + ": " + "string_return_test passed"); 
    else
      System.out.println(LG__() + LINE__() + ": " + "string_return_test failed");
  }
}
