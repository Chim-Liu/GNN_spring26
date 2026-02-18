# Semester Project: Encrypted MNIST Inference using Convolutional Neural Networks and Fully Homomorphic Encryption (FHE)

## Part 0: Project Overview
**Student Name:** Yanchen Liu  
**Course:** 60868  

This project explores Privacy-Preserving Machine Learning (PPML) by combining Convolutional Neural Networks (CNNs) with Fully Homomorphic Encryption (FHE). The primary objective is to build a machine learning pipeline capable of classifying handwritten digits (the MNIST dataset) directly on encrypted image data. This ensures that the server processing the image never sees the raw, plaintext pixels, offering a foundational blueprint for secure, cloud-based image recognition.

## Part 1: Problem Description
As machine learning models increasingly move to cloud servers for inference, data privacy has become a critical concern. In fields like healthcare (medical imaging) or biometric security (facial recognition), clients cannot safely send unencrypted images to third-party servers due to privacy regulations and the risk of data breaches. 

The core problem this project addresses is: **How can a remote server perform spatial feature extraction and classification on an image without ever decrypting the image itself?**

Fully Homomorphic Encryption (FHE) offers a mathematical solution by allowing computations to be performed directly on ciphertexts. However, FHE introduces severe constraints on the types of mathematical operations a neural network can perform. This project will serve as a classic PPML starter exercise, focusing on overcoming these constraints in a controlled environment using regular, grid-structured image data rather than highly complex, irregular graph structures.

## Part 2: Proposed High-Level Solution
The proposed solution implements a secure Client-Server architecture utilizing a modified 2-layer Convolutional Neural Network.

### Phase 1: Plaintext Training and Cryptographic Adaptation
Traditional neural networks rely heavily on non-linear activation functions like ReLU ($f(x) = \max(0, x)$) to learn complex patterns. However, FHE schemes generally only support basic addition and multiplication. Therefore, a standard CNN cannot be directly encrypted.

To solve this, I will first train a lightweight 2-layer CNN on plaintext data. During this architectural design phase, I will replace all non-FHE-friendly activation functions (like ReLU or Sigmoid) with low-order polynomial approximations. Specifically, I will use a simple square activation function ($f(x) = x^2$). Because squaring only requires multiplication, it is fully compatible with FHE circuits. 

### Phase 2: Encrypted Inference Protocol
Once the adapted CNN is trained and its weights are frozen, the inference pipeline will operate as follows:
1.  **Encryption:** The client takes a 28x28 grayscale MNIST image, encrypts it using their private FHE key, and sends the resulting ciphertext to the server.
2.  **Encrypted Computation:** The server loads the pre-trained, unencrypted model weights. It then performs the necessary convolutions, polynomial activations, and pooling operations entirely on the encrypted client data. The server processes this without knowing the contents of the image.
3.  **Decryption:** The server returns an encrypted vector of logits (the predictions for digits 0-9) to the client. The client uses their private key to decrypt the vector and determine the final classification.

## Part 3: Learning Goals and Technical Challenges
Because it is early in the semester, I am focusing on what I need to learn to make this architecture viable:

1.  **Understanding Polynomial Approximations:** I need to learn how replacing ReLU with a square function impacts the gradient descent process during training. Polynomials can cause exploding gradients, so I must research proper weight initialization and normalization techniques suited for square activations.
2.  **FHE Frameworks:** I will need to learn how to bridge standard PyTorch models with cryptographic libraries. I plan to evaluate tools like **TenSEAL** or **Concrete-ML** to determine which offers the most efficient cryptographic inference for CNNs.
3.  **Managing the Noise Budget:** Every multiplication operation in FHE introduces noise into the ciphertext. If the circuit is too deep, the noise overwhelms the data, making decryption impossible. I need to learn how to calculate and manage the "multiplicative depth" of my 2-layer CNN to ensure it successfully compiles without requiring computationally devastating "bootstrapping" techniques.

## Part 4: Datasets and Data Handling
I will use the classic **MNIST Database of Handwritten Digits**. This dataset contains 70,000 grayscale images of digits (0-9), normalized to fit into a 28x28 pixel bounding box. MNIST is ideal because its regular grid structure simplifies the convolution process, allowing me to focus entirely on the cryptographic bottlenecks rather than complex computer vision problems.

To ensure rigorous evaluation and prevent data leakage, the dataset will be partitioned into three strict subsets:

1.  **Training Set (Plaintext):** This subset will be used to train the initial neural network. It will allow the model to learn the spatial features of the digits and optimize the trainable parameters (weights and biases) while using the modified square activation functions.
2.  **Validation Set (Interim Tuning):** This subset is crucial for bridging the plaintext and encrypted worlds. After training the plaintext model, I will use the validation set to test the FHE compilation. I will evaluate how quantization (converting floating-point weights to integers for FHE) impacts the model. If the accuracy drops significantly during the simulated encryption phase, I will use this validation data to adjust the quantization bit-width and the scaling factors of the ciphertext.
3.  **Test Set (The "Unknown" Data):** This subset will remain completely untouched until the final evaluation phase. Once the FHE pipeline is fully built, I will run this set through the *encrypted inference* protocol. This will provide the final metrics for my report.

## Part 5: Evaluation Metrics
The ultimate success of this project will be judged on three comparative metrics derived from the final Test Set:
* **Plaintext Accuracy:** The baseline classification accuracy of the model without encryption.
* **Encrypted Accuracy:** The classification accuracy of the model operating on FHE ciphertexts. A successful project will see little to no degradation between plaintext and encrypted accuracy.
* **Inference Latency:** The time it takes for the server to process one encrypted image. I will measure this to demonstrate the computational overhead introduced by FHE compared to standard inference.

## Part 6: Tools and Libraries
* **Deep Learning Framework:** PyTorch or TensorFlow (for initial model creation).
* **Cryptographic Framework:** TenSEAL (Microsoft SEAL wrapper) or Concrete-ML (Zama).
* **Data Processing:** NumPy, TorchVision.
