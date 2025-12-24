#ifndef __diffcoal_utils_helpers_hpp__
#define __diffcoal_utils_helpers_hpp__

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>

namespace diffcoal
{
    using namespace torch::indexing;

    // Utility class to standardize tensor device and dtype conversion.
    struct DCTensorSpec
    {
    public:
        torch::Device device = torch::kCPU;
        torch::Dtype dtype = torch::kFloat32;

        DCTensorSpec(torch::Device dev = torch::kCPU, torch::Dtype dt = torch::kFloat32)
        : device(dev)
        , dtype(dt)
        {
        }

        DCTensorSpec(const std::string & dev_str, const std::string & dtype_str = "float")
        : device(torch::kCPU)
        , dtype(torch::kFloat32)
        {
            if (!dev_str.empty())
            {
                device = torch::Device(dev_str);
            }

            dtype = stringToDtype(dtype_str);
        }

        torch::Tensor to(const torch::Tensor & x) const
        {
            return x.to(device, dtype);
        }

        template<typename T>
        torch::Tensor to(const std::vector<T> & x) const
        {
            auto opts = torch::TensorOptions().dtype(dtype).device(device);
            return torch::tensor(x, opts);
        }

        template<typename T>
        torch::Tensor to(std::initializer_list<T> x) const
        {
            auto options = torch::TensorOptions().device(device).dtype(dtype);
            return torch::tensor(x, options);
        }

        torch::Tensor toIdx(const torch::Tensor & x) const
        {
            return x.to(device, torch::kLong);
        }

        template<typename T>
        torch::Tensor toIdx(const std::vector<T> & x) const
        {
            auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
            return torch::tensor(x, options);
        }

        template<typename T>
        torch::Tensor toIdx(std::initializer_list<T> x) const
        {
            auto options = torch::TensorOptions().device(device).dtype(torch::kLong);
            return torch::tensor(x, options);
        }

    private:
        torch::Dtype stringToDtype(const std::string & str)
        {
            if (str == "float" || str == "float32")
                return torch::kFloat32;
            if (str == "double" || str == "float64")
                return torch::kFloat64;
            if (str == "half" || str == "float16")
                return torch::kFloat16;
            if (str == "int" || str == "int32")
                return torch::kInt32;
            if (str == "long" || str == "int64")
                return torch::kInt64;
            if (str == "bool")
                return torch::kBool;

            std::cerr << "Warning: Unknown dtype string '" << str << "', defaulting to float32." << std::endl;
            return torch::kFloat32;
        }
    };

    torch::Tensor torchNormalizeVector(torch::Tensor v)
    {
        return v / torch::clamp(v.norm(2, /*dim=*/-1, /*keepdim=*/true), /*min=*/1e-12);
    }

    torch::Tensor skewSymmetric(torch::Tensor v)
    {
        // v: (..., 3)
        auto parts = v.unbind(-1);
        auto v1 = parts[0];
        auto v2 = parts[1];
        auto v3 = parts[2];
        auto O = torch::zeros_like(v1);

        // Row 1: [0, -v3, v2]
        auto r1 = torch::stack({O, -v3, v2}, -1);
        // Row 2: [v3, 0, -v1]
        auto r2 = torch::stack({v3, O, -v1}, -1);
        // Row 3: [-v2, v1, 0]
        auto r3 = torch::stack({-v2, v1, O}, -1);

        return torch::stack({r1, r2, r3}, -2);
    }

} // namespace diffcoal

#endif // __diffcoal_utils_helpers_hpp__
