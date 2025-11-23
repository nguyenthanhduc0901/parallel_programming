#include "cpu/autoencoder.h"
#include <iostream>
#include <fstream>

Autoencoder::Autoencoder() {
    build_model();
}

Autoencoder::~Autoencoder() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void Autoencoder::build_model() {
    // Encoder (lighter CPU baseline)
    // Input: (3, 32, 32)
    layers.push_back(new Conv2D(3, 64, 3, 1, 1));
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2D(2, 2));

    layers.push_back(new Conv2D(64, 32, 3, 1, 1));
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2D(2, 2));
    // Latent: (32, 8, 8)

    // Decoder (mirror of encoder)
    layers.push_back(new Conv2D(32, 32, 3, 1, 1));
    layers.push_back(new ReLU());
    layers.push_back(new UpSample2D(2));

    layers.push_back(new Conv2D(32, 64, 3, 1, 1));
    layers.push_back(new ReLU());
    layers.push_back(new UpSample2D(2));

    layers.push_back(new Conv2D(64, 3, 3, 1, 1));
    // Output: (3, 32, 32)
}

Tensor Autoencoder::forward(const Tensor& input) {
    Tensor x = input;
    for (Layer* layer : layers) {
        x = layer->forward(x);
    }
    return x;
}

float Autoencoder::backward(const Tensor& input, const Tensor& output, float learning_rate) {
    // Compute MSE Loss Gradient: 2 * (Output - Target) / N
    // Here Target is Input (Reconstruction)
    Tensor grad(output.b, output.c, output.h, output.w);
    const int N = output.size();

    float loss = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float diff = output.data[i] - input.data[i];
        loss += diff * diff;
        grad.data[i] = 2.0f * diff / output.b; // Normalize by batch size
    }
    loss /= static_cast<float>(N);

    // Backpropagate
    Tensor cur_grad = grad;
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        cur_grad = layers[i]->backward(cur_grad, learning_rate);
    }
    return loss;
}

std::vector<float> Autoencoder::extract_features(const Tensor& input) {
    Tensor x = input;
    // Run through encoder layers (indices 0 to 5)
    // 0: Conv, 1: ReLU, 2: MaxPool, 3: Conv, 4: ReLU, 5: MaxPool
    for (int i = 0; i <= 5; ++i) {
        x = layers[i]->forward(x);
    }
    // Flatten
    return x.data;
}

void Autoencoder::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for saving: " << filepath << std::endl;
        return;
    }
    for (Layer* layer : layers) {
        layer->save(file);
    }
    file.close();
    std::cout << "Model saved to " << filepath << std::endl;
}

void Autoencoder::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for loading: " << filepath << std::endl;
        return;
    }
    for (Layer* layer : layers) {
        layer->load(file);
    }
    file.close();
    std::cout << "Model loaded from " << filepath << std::endl;
}
