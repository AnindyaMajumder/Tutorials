# Setting Up a Deep Learning Environment with GPU Support on Ubuntu 24.04

This guide will walk you through the process of setting up a GPU-accelerated deep learning environment on Ubuntu 24.04. Each command is explained in detail to help beginners understand the purpose of every step.

**Note:** While both PyTorch and TensorFlow can be installed in the same environment, it is recommended to use separate virtual environments for each framework to avoid version conflicts or unexpected behavior.

---

## Why Use a GPU for Deep Learning?
GPUs are specifically designed for parallel processing, which significantly speeds up the training and inference of deep learning models. Setting up a proper environment with GPU support ensures you make the most of your hardware's capabilities.

---

## Prerequisites
Before we start, ensure:
1. You have an NVIDIA GPU installed.
2. You have administrator (sudo) privileges on your system.
3. Your system is connected to the internet.

---

## 1. Check GPU Compatibility
```bash
lspci | grep -i nvidia
```
This command lists all NVIDIA devices connected to your system. It ensures your GPU is detected and compatible with NVIDIA drivers.

---

## 2. Confirm Your System Architecture and OS Version
```bash
uname -m && cat /etc/*release
```
This command tells us the system architecture (e.g., `x86_64`) and the version of Ubuntu you're running. This information is crucial for downloading the correct NVIDIA drivers and CUDA version.

---

## 3. Check GCC and G++ Compiler Versions
```bash
gcc --version
g++ --version
```
CUDA requires compatible versions of GCC and G++. These commands check the installed versions to ensure compatibility with CUDA.

If GCC or G++ is missing or outdated, install them:
```bash
sudo apt install gcc g++ --fix-missing
```
This command installs both GCC and G++ compilers and resolves any missing dependencies.

---

## 4. Update and Upgrade Your System
```bash
sudo apt update && sudo apt upgrade -y
```
This updates the package list and upgrades installed packages to the latest versions, ensuring system stability and compatibility with the latest drivers.

---

## 5. Install NVIDIA Drivers
```bash
sudo ubuntu-drivers install
```
This command automatically identifies the best NVIDIA driver for your GPU and installs it.

After installation, clean up unnecessary packages:
```bash
sudo apt autoremove
```
This removes unused dependencies, freeing up space.

Verify the installation:
```bash
nvidia-smi
```
This command confirms that NVIDIA drivers are correctly installed and displays information about your GPU.

---

## 6. Install CUDA Toolkit
### Why CUDA?
CUDA is NVIDIA's platform for parallel computing. It allows you to use the GPU for general-purpose processing, which is essential for deep learning frameworks like PyTorch and TensorFlow.

### Steps to Install CUDA
1. Visit the [CUDA Downloads Page](https://developer.nvidia.com/cuda-downloads) to select the appropriate version for your system.

2. Add the CUDA repository pin:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

3. Download the CUDA installer:
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
```

4. Install the downloaded package:
```bash
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
```

5. Add the repository key:
```bash
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
```

6. Update the package list:
```bash
sudo apt-get update
```

7. Install the CUDA toolkit:
```bash
sudo apt-get -y install cuda-toolkit-12-8
```

**Installed CUDA Version:** 12.8

---

## 7. Choose Between `cuda-drivers` or `nvidia-open`
You can install either of these based on your preference:
- Use `cuda-drivers` for traditional NVIDIA drivers:
  ```bash
  sudo apt-get install -y cuda-drivers
  ```
- Use `nvidia-open` for open-source NVIDIA drivers:
  ```bash
  sudo apt-get install -y nvidia-open
  ```

---

## 8. Install Python and Pip
Check if Python is installed:
```bash
python3 --version
```

If Python is missing, install it:
```bash
sudo apt install python3
```

Install Pip (Python's package manager):
```bash
sudo apt install python3-pip
```

**Installed Python Version:** 3.12.3

---

## 9. Set Up a Python Virtual Environment
1. Create a project directory:
```bash
mkdir Playground
cd Playground/
```

2. Install the Python virtual environment package:
```bash
sudo apt install python3.12-venv
```

3. Create a virtual environment:
```bash
python3 -m venv .venv
```

4. Activate the virtual environment:
```bash
source .venv/bin/activate
```

---

## 10. Install PyTorch with GPU Support
Inside the virtual environment, install PyTorch:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Installed PyTorch Version:** 12.6

---

## 11. Install TensorFlow with GPU Support
TensorFlow is another popular deep learning framework. It is ideal to intall Tensorflow and Pytorch into different environment. <br/>
Visit the [Tensorflow Download Page](https://www.tensorflow.org/install) to select the appropriate version for your system.
Install TensorFlow with GPU and CUDA support by running the following command inside your virtual environment:

```bash
pip install 'tensorflow[and-cuda]'
```

### Verify TensorFlow GPU Support
After installation, verify that TensorFlow detects your GPU:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If TensorFlow is installed correctly and the GPU is recognized, the output will list your GPU device(s).

**Installed TensorFlow Version:** 2.10

---

## 12. Install Essential Libraries for Deep Learning, Machine Learning, and Computer Vision (Optional or When Necessary)
Enhance your environment by installing popular libraries for deep learning, machine learning, and computer vision.

### Install Core Libraries
```bash
pip3 install numpy scipy pandas matplotlib seaborn scikit-learn
```
- `numpy`: For numerical computations.
- `scipy`: For scientific computing.
- `pandas`: For data manipulation and analysis.
- `matplotlib` & `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms and preprocessing tools.

### Install Computer Vision Libraries
```bash
pip3 install opencv-python pillow
```
- `opencv-python`: For computer vision tasks.
- `pillow`: For image processing.

### Install Jupyter Notebook
```bash
pip3 install jupyterlab
```
- Jupyter is an interactive notebook environment for running and visualizing code.

---

## 13. Verify CUDA with PyTorch
To examine if CUDA is working properly, use this script:

```bash
(.venv) username@PCname:~$ python3
Python 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> exit()
```

You can also test the speed of matrix multiplications on both CPU and GPU using the following script:

```python
import time
import torch

if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cpu")
print("using", device, "device")

matrix_size = 30*512

x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)

print("************* CPU SPEED *******************")
start = time.time()
result = torch.matmul(x, y)
print(time.time() - start)
print("verify device:", result.device)

x_gpu = x.to(device)
y_gpu = y.to(device)
torch.cuda.synchronize()

for i in range(3):
   print("************* GPU SPEED *******************")
   start = time.time()
   result_gpu = torch.matmul(x_gpu, y_gpu)
   torch.cuda.synchronize()
   print(time.time() - start)
   print("verify device:", result_gpu.device)
```

### Expected Output
- For `torch.cuda.is_available()`, the output should be `True`.
- The script will demonstrate a significant speed-up for GPU operations compared to CPU.

**Script Reference:** [CUDA on WSL2 by FahimFBA](https://fahimfba.github.io/CUDA-WSL2-Ubuntu/#/?id=step-13-test)

---

## Conclusion
Congratulations! You have successfully set up a deep learning environment with GPU support on Ubuntu 24.04. You can now utilize your GPU to accelerate deep learning tasks with frameworks like PyTorch and TensorFlow. Additionally, the installed libraries equip you for data analysis, machine learning, and computer vision tasks.

Enjoy your deep learning journey!
