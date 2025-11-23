#include "common/cifar10_loader.h"
#include "cpu/autoencoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>

// Hyperparameters
const int BATCH_SIZE = 32;
const int EPOCHS = 1; // Reduced for testing, project asks for 20
const float LEARNING_RATE = 0.001f;

void train(Autoencoder& model, const std::vector<Image>& data) {
    int num_samples = data.size();
    int num_batches = (num_samples + BATCH_SIZE - 1) / BATCH_SIZE;
    
    std::vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) indices[i] = i;
    
    std::default_random_engine rng(std::random_device{}());

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float total_loss = 0.0f;
        
        for (int b = 0; b < num_batches; ++b) {
            int start_idx = b * BATCH_SIZE;
            int end_idx = std::min(start_idx + BATCH_SIZE, num_samples);
            int current_batch_size = end_idx - start_idx;
            
            // Prepare batch tensor
            Tensor input_batch(current_batch_size, 3, 32, 32);
            for (int i = 0; i < current_batch_size; ++i) {
                const Image& img = data[indices[start_idx + i]];
                // Copy data to tensor (CHW format)
                // Image data is already flat 3072 (R...G...B...)
                // Tensor expects flat (B...C...H...W...)
                // We need to copy img.data into input_batch.data at the right offset
                std::copy(img.data.begin(), img.data.end(), input_batch.data.begin() + i * 3072);
            }
            
            // Forward
            Tensor output = model.forward(input_batch);

            // Backward (returns MSE loss for this batch)
            float batch_loss = model.backward(input_batch, output, LEARNING_RATE);
            total_loss += batch_loss;
            
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                        << " Batch " << b + 1 << "/" << num_batches 
                        << " Loss: " << batch_loss << "\r" << std::flush;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "\nEpoch " << epoch + 1 << " completed in " << elapsed.count() << "s. "
                  << "Avg Loss: " << total_loss / num_batches << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string data_dir = "data"; // Default
    if (argc > 1) data_dir = argv[1];

    // 1. Load Data
    Cifar10Loader loader(data_dir);
    loader.load_data();
    
    const auto& train_data = loader.get_train_images();
    if (train_data.empty()) {
        std::cerr << "No training data found. Please check data directory." << std::endl;
        return 1;
    }

    // 2. Init Model
    std::cout << "Initializing Autoencoder..." << std::endl;
    Autoencoder model;

    // 3. Train
    std::cout << "Starting training on CPU..." << std::endl;
    train(model, train_data);

    std::cout << "Training complete." << std::endl;
    return 0;
}
