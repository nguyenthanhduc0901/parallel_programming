#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <string>
#include <vector>
#include <iostream>

struct Image {
    std::vector<float> data; // Normalized pixel values [0, 1] (3 * 32 * 32)
    int label;
};

class Cifar10Loader {
public:
    Cifar10Loader(const std::string& data_dir);
    void load_data();
    
    const std::vector<Image>& get_train_images() const { return train_images; }
    const std::vector<Image>& get_test_images() const { return test_images; }

private:
    std::string data_dir;
    std::vector<Image> train_images;
    std::vector<Image> test_images;

    void read_batch(const std::string& filename, std::vector<Image>& dataset);
};

#endif // CIFAR10_LOADER_H
