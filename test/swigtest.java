package test.swig;

public class swigtest {
  static {
    System.loadLibrary("swigtest");
  }

  public static void main(String argv[]) {
    System.out.println(test.fact(4));
  }
}
