#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "cpu/layers.h"
#include <vector>
#include <memory>

class Autoencoder {
public:
    Autoencoder();
    ~Autoencoder();

    Tensor forward(const Tensor& input);
    // Performs backward pass and returns MSE loss for the given batch
    float backward(const Tensor& input, const Tensor& output, float learning_rate);
    
    // Extract features from the latent space (output of encoder)
    std::vector<float> extract_features(const Tensor& input);

    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);

private:
    std::vector<Layer*> layers;
    
    // Helper to add layers
    void build_model();
};

#endif // AUTOENCODER_H
