#pragma once

#include <torch/torch.h>

// Anonymous namespace to bound the scope of using in this file
namespace
{
    using namespace torch;
};

void initialize_weights(torch::nn::Module &module)
{
    for (auto &submodule : module.children())
    {
        if (auto conv = dynamic_cast<torch::nn::Conv2dImpl *>(submodule.get()))
        {
            torch::nn::init::xavier_uniform_(conv->weight);
            if (conv->bias.defined())
            {
                torch::nn::init::constant_(conv->bias, 0);
            }
        }
        else if (auto trans_conv = dynamic_cast<torch::nn::ConvTranspose2dImpl *>(submodule.get()))
        {
            torch::nn::init::xavier_uniform_(trans_conv->weight);
            if (trans_conv->bias.defined())
            {
                torch::nn::init::constant_(trans_conv->bias, 0);
            }
        }
        else
        {
            // Recursively initialize weights for nested modules
            initialize_weights(*submodule);
        }
    }
}

// ConvBlock class
struct ConvBlockImpl : torch::nn::Module
{
    torch::nn::Conv2d conv{nullptr};
    torch::nn::ReLU relu;

    ConvBlockImpl(int64_t in_channels, int64_t out_channels)
        : conv(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)),
          relu(torch::nn::ReLUOptions().inplace(true))
    {
        register_module("conv", conv);
        register_module("relu", relu);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto y = conv->forward(x);
        return relu->forward(y);
    }
};

TORCH_MODULE(ConvBlock);

// UNetSimple class
struct UNetImpl : torch::nn::Module
{
    // Encoding path
    ConvBlock c1_1{nullptr}, c1_2{nullptr};
    torch::nn::MaxPool2d pool1;

    ConvBlock c2_1{nullptr}, c2_2{nullptr};
    torch::nn::MaxPool2d pool2;

    ConvBlock c3_1{nullptr}, c3_2{nullptr};
    torch::nn::ConvTranspose2d upconv3;

    // Decoding path
    ConvBlock c4_1{nullptr}, c4_2{nullptr};
    torch::nn::ConvTranspose2d upconv4;

    ConvBlock c5_1{nullptr};

    UNetImpl(int in_channels, int out_channels)
        : c1_1(ConvBlock(in_channels, 8)),
          c1_2(ConvBlock(8, 8)),
          pool1(torch::nn::MaxPool2dOptions(2)),

          c2_1(ConvBlock(8, 16)),
          c2_2(ConvBlock(16, 16)),
          pool2(torch::nn::MaxPool2dOptions(2)),

          c3_1(ConvBlock(16, 32)),
          c3_2(ConvBlock(32, 32)),
          upconv3(torch::nn::ConvTranspose2dOptions(32, 32, 1).stride(2).padding(0).output_padding(1)),

          c4_1(ConvBlock(32 + 16, 16)),
          c4_2(ConvBlock(16, 16)),
          upconv4(torch::nn::ConvTranspose2dOptions(16, 16, 1).stride(2).padding(0).output_padding(1)),

          c5_1(ConvBlock(16 + 8, out_channels))
    {
        // Register modules
        register_module("c1_1", c1_1);
        register_module("c1_2", c1_2);
        register_module("pool1", pool1);

        register_module("c2_1", c2_1);
        register_module("c2_2", c2_2);
        register_module("pool2", pool2);

        register_module("c3_1", c3_1);
        register_module("c3_2", c3_2);
        register_module("upconv3", upconv3);

        register_module("c4_1", c4_1);
        register_module("c4_2", c4_2);
        register_module("upconv4", upconv4);

        register_module("c5_1", c5_1);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto c1_1_out = c1_1->forward(x);
        auto c1_2_out = c1_2->forward(c1_1_out);
        auto c1_3 = pool1->forward(c1_2_out);

        auto c2_1_out = c2_1->forward(c1_3);
        auto c2_2_out = c2_2->forward(c2_1_out);
        auto c2_3 = pool2->forward(c2_2_out);

        auto c3_1_out = c3_1->forward(c2_3);
        auto c3_2_out = c3_2->forward(c3_1_out);
        auto c3_3 = upconv3->forward(c3_2_out);

        // Concatenate along channel dimension (dim=1)
        auto c4_in = torch::cat({c3_3, c2_2_out}, 1);
        auto c4_1_out = c4_1->forward(c4_in);
        auto c4_2_out = c4_2->forward(c4_1_out);
        auto c4_3 = upconv4->forward(c4_2_out);

        auto c5_in = torch::cat({c4_3, c1_2_out}, 1);
        auto c5_1_out = c5_1->forward(c5_in);

        return c5_1_out;
    }
};
TORCH_MODULE(UNet);

class MainNetImpl : public torch::nn::Module
{
public:
    MainNetImpl()
        : mUnet(15, 8),
          mParamConv(torch::nn::Conv2dOptions(8, 6, 1))
    {
        register_module("mUnet", mUnet);
        register_module("mParamConv", mParamConv);

        initialize_weights(*this);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor netColor1, torch::Tensor netColor2, torch::Tensor netPrev, torch::Tensor netAlbedo, torch::Tensor netNormal)
    {
        // Concatenate inputs along channel dimension
        torch::Tensor net_input = torch::cat({netColor1, netColor2, netPrev, netAlbedo, netNormal}, 1);

        // Pass through UNet
        torch::Tensor net_out = mUnet->forward(net_input);

        // Get final output layers
        torch::Tensor param = mParamConv->forward(net_out);
        torch::Tensor band, alpha;
        auto ret = torch::split(param, {5, 1}, 1);
        band = ret[0];
        alpha = ret[1];

        band = 1 / (torch::square(band) + 1e-4);
        alpha = torch::sigmoid(alpha);

        return {band, alpha};
    }

private:
    UNet mUnet{nullptr};
    torch::nn::Conv2d mParamConv{nullptr};
};
TORCH_MODULE(MainNet);