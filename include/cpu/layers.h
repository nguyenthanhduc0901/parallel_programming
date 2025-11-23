#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <cmath>
#include <random>
#include <fstream>

#include <stdexcept>
#include <iostream>

struct Tensor {
    std::vector<float> data;
    int b, c, h, w;

    Tensor(int b = 1, int c = 1, int h = 1, int w = 1) : b(b), c(c), h(h), w(w) {
        data.resize(b * c * h * w, 0.0f);
    }
    
    float& at(int ib, int ic, int ih, int iw) {
        int index = ((ib * c + ic) * h + ih) * w + iw;
        if (index < 0 || index >= data.size()) {
            std::cerr << "Tensor access out of bounds! Index: " << index << " Size: " << data.size() 
                      << " (b=" << ib << ", c=" << ic << ", h=" << ih << ", w=" << iw << ")"
                      << " TensorShape(" << b << "," << c << "," << h << "," << w << ")" << std::endl;
            throw std::out_of_range("Tensor index out of range");
        }
        return data[index];
    }

    const float& at(int ib, int ic, int ih, int iw) const {
        int index = ((ib * c + ic) * h + ih) * w + iw;
        if (index < 0 || index >= data.size()) {
            std::cerr << "Tensor access out of bounds! Index: " << index << " Size: " << data.size() 
                      << " (b=" << ib << ", c=" << ic << ", h=" << ih << ", w=" << iw << ")"
                      << " TensorShape(" << b << "," << c << "," << h << "," << w << ")" << std::endl;
            throw std::out_of_range("Tensor index out of range");
        }
        return data[index];
    }
    
    int size() const { return b * c * h * w; }
};

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output, float learning_rate) = 0;
    virtual void save(std::ofstream& file) const {}
    virtual void load(std::ifstream& file) {}
    virtual ~Layer() {}
};

class Conv2D : public Layer {
public:
    Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 1);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
    void save(std::ofstream& file) const override;
    void load(std::ifstream& file) override;

private:
    int in_c, out_c, k_size, stride, padding;
    Tensor weights;
    Tensor biases;
    Tensor input_cache; // Store input for backward pass
    
    void init_weights();
};

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
private:
    Tensor input_cache;
};

class MaxPool2D : public Layer {
public:
    MaxPool2D(int pool_size = 2, int stride = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
private:
    int pool_size, stride;
    Tensor input_cache;
    std::vector<int> max_indices; // Store indices for backward pass
};

class UpSample2D : public Layer {
public:
    UpSample2D(int scale_factor = 2);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float learning_rate) override;
private:
    int scale_factor;
    Tensor input_cache; // Store input shape
};

#endif // LAYERS_H
