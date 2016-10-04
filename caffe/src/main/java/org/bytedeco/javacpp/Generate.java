package org.bytedeco.javacpp;

public class Generate {
  public static void main(String[] args) {
    Class preset = org.bytedeco.javacpp.presets.americano.class;

    String java = "src/main/java";
    JavaCPP.generateJava(preset, java);
    JavaCPP.compileJava(preset, java);

    String cpp = "src";
    JavaCPP.generateCpp(preset, cpp);
    JavaCPP.compileCpp(preset, cpp, false);
  }
}

