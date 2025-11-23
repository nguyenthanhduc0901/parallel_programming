#include "common/cifar10_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>

Cifar10Loader::Cifar10Loader(const std::string& dir) : data_dir(dir) {}

void Cifar10Loader::read_batch(const std::string& filename, std::vector<Image>& dataset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // CIFAR-10 binary format: <1 x label><3072 x pixel>
    // 3072 bytes = 32x32 red, 32x32 green, 32x32 blue
    const int record_size = 1 + 3072;
    
    while (file.peek() != EOF) {
        std::vector<unsigned char> buffer(record_size);
        file.read(reinterpret_cast<char*>(buffer.data()), record_size);
        
        if (file.gcount() != record_size) break;

        Image img;
        img.label = buffer[0];
        img.data.resize(3072);

        // Normalize to [0, 1]
        for (int i = 0; i < 3072; ++i) {
            img.data[i] = static_cast<float>(buffer[i + 1]) / 255.0f;
        }
        dataset.push_back(img);
    }
    file.close();
}

void Cifar10Loader::load_data() {
    std::cout << "Loading CIFAR-10 data from " << data_dir << "..." << std::endl;
    
    // Load 5 training batches
    for (int i = 1; i <= 5; ++i) {
        std::string filename = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        read_batch(filename, train_images);
    }

    // Load test batch
    read_batch(data_dir + "/test_batch.bin", test_images);

    std::cout << "Loaded " << train_images.size() << " training images." << std::endl;
    std::cout << "Loaded " << test_images.size() << " test images." << std::endl;
}
