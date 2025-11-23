#include "cpu/layers.h"
#include <iostream>
#include <random>

// --- Conv2D ---
Conv2D::Conv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_c(in_channels), out_c(out_channels), k_size(kernel_size), stride(stride), padding(padding) {
    weights = Tensor(1, out_c, in_c, k_size * k_size); // Simplified storage for weights
    // Actually, weights should be (out_c, in_c, k, k). Let's use the Tensor struct flexibly.
    // We'll store weights as (out_c, in_c, k_size, k_size)
    weights = Tensor(out_c, in_c, k_size, k_size);
    biases = Tensor(1, out_c, 1, 1);
    init_weights();
}

void Conv2D::init_weights() {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 0.1); // Xavier/He init would be better
    for (float& w : weights.data) w = distribution(generator);
    for (float& b : biases.data) b = 0.0f;
}

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;
    int out_h = (input.h + 2 * padding - k_size) / stride + 1;
    int out_w = (input.w + 2 * padding - k_size) / stride + 1;
    Tensor output(input.b, out_c, out_h, out_w);
    // Use direct index arithmetic instead of at() to reduce overhead
    const std::vector<float>& in_data = input.data;
    const std::vector<float>& w_data  = weights.data;
    const std::vector<float>& b_data  = biases.data;
    std::vector<float>& out_data      = output.data;

    const int in_b = input.b;
    const int in_c_ = input.c;
    const int in_h = input.h;
    const int in_w = input.w;

    for (int b = 0; b < in_b; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bias = b_data[oc];
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias;
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            const int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= in_h) continue;
                            for (int kw = 0; kw < k_size; ++kw) {
                                const int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= in_w) continue;

                                const int in_idx = ((b * in_c_ + ic) * in_h + ih) * in_w + iw;
                                const int w_idx  = ((oc * in_c_ + ic) * k_size + kh) * k_size + kw;
                                sum += in_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }
                    const int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output, float learning_rate) {
    // Debug check
    if (grad_output.b != input_cache.b || grad_output.c != out_c) {
        std::cerr << "Conv2D Backward Mismatch: Grad(" << grad_output.b << "," << grad_output.c << "," << grad_output.h << "," << grad_output.w 
                  << ") vs Expected(B," << out_c << ",H,W)" << std::endl;
    }

    // Simplified backward pass (Gradient w.r.t Input and Weights)
    // NOTE: This is a very computationally expensive naive implementation.
    Tensor grad_input(input_cache.b, input_cache.c, input_cache.h, input_cache.w);
    Tensor grad_weights(weights.b, weights.c, weights.h, weights.w); // zero-initialized
    Tensor grad_biases(biases.b, biases.c, biases.h, biases.w);     // zero-initialized

    std::vector<float>& g_in  = grad_input.data;
    std::vector<float>& g_w   = grad_weights.data;
    std::vector<float>& g_b   = grad_biases.data;
    const std::vector<float>& in_data  = input_cache.data;
    const std::vector<float>& w_data   = weights.data;
    const std::vector<float>& go_data  = grad_output.data;

    const int batch = grad_output.b;
    const int out_h = grad_output.h;
    const int out_w = grad_output.w;
    const int in_h  = input_cache.h;
    const int in_w  = input_cache.w;

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    const int go_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    const float grad = go_data[go_idx];
                    g_b[oc] += grad;

                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int kh = 0; kh < k_size; ++kh) {
                            const int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= in_h) continue;
                            for (int kw = 0; kw < k_size; ++kw) {
                                const int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= in_w) continue;

                                const int in_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                                const int w_idx  = ((oc * in_c + ic) * k_size + kh) * k_size + kw;

                                g_w[w_idx] += in_data[in_idx] * grad;
                                g_in[in_idx] += w_data[w_idx] * grad;
                            }
                        }
                    }
                }
            }
        }
    }

    // Update weights and biases
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] -= learning_rate * g_w[i];
    }
    for (size_t i = 0; i < biases.data.size(); ++i) {
        biases.data[i] -= learning_rate * g_b[i];
    }

    return grad_input;
}

void Conv2D::save(std::ofstream& file) const {
    // Save weights
    file.write(reinterpret_cast<const char*>(weights.data.data()), weights.data.size() * sizeof(float));
    // Save biases
    file.write(reinterpret_cast<const char*>(biases.data.data()), biases.data.size() * sizeof(float));
}

void Conv2D::load(std::ifstream& file) {
    // Load weights
    file.read(reinterpret_cast<char*>(weights.data.data()), weights.data.size() * sizeof(float));
    // Load biases
    file.read(reinterpret_cast<char*>(biases.data.data()), biases.data.size() * sizeof(float));
}

// --- ReLU ---
Tensor ReLU::forward(const Tensor& input) {
    input_cache = input;
    Tensor output = input;
    for (float& val : output.data) {
        if (val < 0) val = 0;
    }
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        if (input_cache.data[i] <= 0) grad_input.data[i] = 0;
    }
    return grad_input;
}

// --- MaxPool2D ---
MaxPool2D::MaxPool2D(int pool_size, int stride) : pool_size(pool_size), stride(stride) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    input_cache = input;
    int out_h = (input.h - pool_size) / stride + 1;
    int out_w = (input.w - pool_size) / stride + 1;
    Tensor output(input.b, input.c, out_h, out_w);
    max_indices.resize(output.size());

    int idx = 0;
    for (int b = 0; b < input.b; ++b) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -1e9;
                    int max_idx = -1;
                    
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            // Check bounds for ih/iw (though logic should guarantee it)
                            if (ih < input.h && iw < input.w) {
                                float val = input.at(b, c, ih, iw);
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = ((b * input.c + c) * input.h + ih) * input.w + iw;
                                }
                            }
                        }
                    }
                    output.at(b, c, oh, ow) = max_val;
                    if (max_idx == -1) {
                         // Fallback if all values were somehow skipped (should not happen with valid logic)
                         // Assign to first element of window to avoid crash
                         int ih = oh * stride;
                         int iw = ow * stride;
                         max_idx = ((b * input.c + c) * input.h + ih) * input.w + iw;
                    }
                    max_indices[idx++] = max_idx;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output, float learning_rate) {
    if (grad_output.size() != max_indices.size()) {
        std::cerr << "MaxPool2D Backward Mismatch: GradSize " << grad_output.size() << " vs IndicesSize " << max_indices.size() << std::endl;
    }
    Tensor grad_input(input_cache.b, input_cache.c, input_cache.h, input_cache.w); // Init to 0
    
    int idx = 0;
    for (int i = 0; i < grad_output.size(); ++i) {
        int max_idx = max_indices[i];
        if (max_idx >= 0 && max_idx < grad_input.data.size()) {
            grad_input.data[max_idx] += grad_output.data[i];
        } else {
             std::cerr << "MaxPool2D Backward: Invalid max_idx " << max_idx << " at i=" << i << std::endl;
        }
    }
    return grad_input;
}

// --- UpSample2D ---
UpSample2D::UpSample2D(int scale_factor) : scale_factor(scale_factor) {}

Tensor UpSample2D::forward(const Tensor& input) {
    input_cache = input;
    Tensor output(input.b, input.c, input.h * scale_factor, input.w * scale_factor);
    
    for (int b = 0; b < input.b; ++b) {
        for (int c = 0; c < input.c; ++c) {
            for (int ih = 0; ih < input.h; ++ih) {
                for (int iw = 0; iw < input.w; ++iw) {
                    float val = input.at(b, c, ih, iw);
                    for (int sh = 0; sh < scale_factor; ++sh) {
                        for (int sw = 0; sw < scale_factor; ++sw) {
                            output.at(b, c, ih * scale_factor + sh, iw * scale_factor + sw) = val;
                        }
                    }
                }
            }
        }
    }
    return output;
}

Tensor UpSample2D::backward(const Tensor& grad_output, float learning_rate) {
    Tensor grad_input(input_cache.b, input_cache.c, input_cache.h, input_cache.w); // Init to 0
    
    for (int b = 0; b < grad_output.b; ++b) {
        for (int c = 0; c < grad_output.c; ++c) {
            for (int oh = 0; oh < grad_output.h; ++oh) {
                for (int ow = 0; ow < grad_output.w; ++ow) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    grad_input.at(b, c, ih, iw) += grad_output.at(b, c, oh, ow);
                }
            }
        }
    }
    return grad_input;
}
