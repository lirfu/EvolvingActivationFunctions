# Procedure for installing Deeplearning4Java library into your project
This is a procedure I found while struggling to install DL4J.
I hope this procedure will help get you to on track swiftly and help you to understand what are you installing.

This procedure was written in `April 2019.`.

## (Preamble) IntelliJ setup
If you want to open this project in IntelliJ you need to set some things up. 
The easiest way is to make a new project in IntelliJ by cloning the repository (`Checkout from version control`).

IntelliJ probably generated its own `.iml` file so you need to tell it where it can find the sources and test files. 
Right clicking on the `src` or `res` directory, under `Mark directory as` select `Sources Root` and `Test Sources Root` respectably.

This project also uses a 'submodule' library for drawing graphs. 
At this point, if you tried to compile it probably got a lot of 'missing import' errors (due to missing libraries).
To get the sources of this library, simply run `git submodule update --init`, find its' directory (probably still called `LirfuGraph`) and mark its `src` directory as source (same as before).


## 1. Install DL4J prerequisites
The most important prerequisite is Maven, as it's used to download all 
DL4J packages and libs.

As dependencies might change over time, you should install them following 
the official 'Quickstart: Prerequisites' guide.
At the time of writting, this was the guide: 
[Quickstart](https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart)

> If you want to use GPUs, make sure you have a 64-bit version of JDK installed. I used OpenJDK 11.0.2 on Windows and OpenJDK 1.8.0_191 on Ubuntu.

## 2. Configure the pom.xml file
This file specifies all the dependencies needed for your project, including DL4J.

Now, this part can get a bit tricky as Windows OS needed some dependencies
 I didn't need on Linux to get it running. 

Common dependencies can, again, be found on the official website: [Quickstart](https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart).

Here are mine:
```xml
<dependencies>
...
<!-- Core lib for DL4J (neural networks and stuff) -->
    <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>deeplearning4j-core</artifactId>
        <version>1.0.0-beta3</version>
    </dependency>
<!-- Core lib for ND4J (matrices, linear algebra and stuff) -->
    <dependency>
        <groupId>org.nd4j</groupId>
        <artifactId>nd4j-native-platform</artifactId>
        <version>1.0.0-beta3</version>
    </dependency>
```

This should be enough to run an example solely on CPU.


## 3. Install CUDA
To utilize GPUs you need to install the CUDA library. You can do this by installing
the CUDA Toolkit: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

> Sometimes you have to additionally manually install your GPU drivers.

On Ubuntu 16.04 I installed version `7.5.17` and on Windows 10 I installed version `10.1.105`.

Follow along the [installation guide](https://docs.nvidia.com/cuda/index.html#installation-guides) and finally check if it installed by checking 
the Nvidia CUDA compiler version:
`nvcc --version`

The guide also offers steps for post-installation and verification.
I definitely recommend you to download, compile and run the mentioned examples. 
It is a lengthy process but it will verify the underlying libraries installed correctly and verify that your card is detected. 

To monitor the state of your GPU you can use `nvidia-smi` (look it up) which I definitely recommend when running your programs.

> Hint: Sometimes you need to restart your computer before usage (on both OSs), because of reasons...


## 4. Install cuDNN
To utilize the pre-made algorithms you need to install the cuDNN library:
[cuDNN Homepage](https://developer.nvidia.com/cudnn)

The installation in my time was just copy/pasting files to appropriate locations in the CUDA directory.

They might require you to register to their website. 
Don't worry, it's free (at least it was in my time).

You should install the version matching the CUDA version installed previously 
and the version supported by DL4J.

On Ubuntu 16.04 I installed version `7.0.5` and on Windows 10 I installed version `7.5.0`.

You can check the cuDNN version by reading the `cudnn.h` header. In the first few `#define` statements 
you can read the version as `MAJOR.MINOR.PATCHLEVEL`. Header is located in the CUDA directory (on Linux find it with `whereis cudnn.h`).

> Hint: Sometimes you need to restart your computer before usage (on both OSs), because of reasons...


## 5. Add DL4J GPU libs
To use CUDA and cuDNN in your DL4J project, you need to add dependencies.
The official guide for utilising CUDA in DL4J is here: [ND4J backends for GPUs and CPUs](https://deeplearning4j.org/docs/latest/deeplearning4j-config-gpu-cpu)
Depending on your CUDA-cuDNN verisons you need to add appropriate libraries.
You can check the official guide for exact supported version pairs:
[Using Deeplearning4j with cuDNN](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn)

On Ubuntu I needed only these:
```xml
<!-- Bridge between DL4J and CUDA. -->
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-cuda-9.0</artifactId>
    <version>1.0.0-beta3</version>
</dependency>
<!-- Bridge between ND4J and CUDA. -->
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-9.0</artifactId>
    <version>1.0.0-beta3</version>
</dependency>
``` 

On Windows however, I needed to add this as well since I'm using a newer version of CUDA
(read in [Using Deeplearning4j with cuDNN](https://deeplearning4j.org/docs/latest/deeplearning4j-config-cudnn)):
```xml
<!-- Additional bridge for newer CUDA version (9.0 an ) -->
<dependency>
    <groupId>org.bytedeco.javacpp-presets</groupId>
    <artifactId>cuda</artifactId>
    <version>10.0-7.3-1.4.3</version>
    <classifier>windows-x86_64-redist</classifier>
</dependency>
```

> Note: Double-check that your `<version>` matches your CUDA version and `<classifier>` matches your OS.

> Hint: Sometimes you need to restart your computer before usage (on both OSs), because of reasons...


## 6. Grab a drink and run your algorithms on GPU!
Now you should be able to utilise the immense power of your GPUs.

If problems occur, double check the compatibilities between DL4J, cuDNN and CUDA.
Also make sure your GPU drivers are installed (in case you bought a new GPU for the occasion).
When all fails, turn to the community and never give up!

Follow along this official quickstart tutorial to test the complete installation process:
[Quickstart for Deeplearning4j](https://deeplearning4j.org/tutorials/00-quickstart-for-deeplearning4j)
