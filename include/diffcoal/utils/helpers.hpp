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

            std::cerr << "Warning: Unknown dtype string '" << str << "', defaulting to float32."
                      << std::endl;
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

    torch::Tensor torchSe3ToMatrixGrad(const torch::Tensor & A, const torch::Tensor & xi)
    {
        auto rho = xi.index({"...", Slice(None, 3)});
        auto phi = xi.index({"...", Slice(3, None)});
        auto skew_rho = skew_symmetric(rho);

        auto G_body = torch::zeros_like(A);
        G_body.index_put_({"...", Slice(None, 3), Slice(None, 3)}, skew_rho);
        G_body.index_put_({"...", Slice(None, 3), 3}, phi);

        return torch::matmul(A, G_body);
    }

    torch::Tensor torchMatrixGradToSe3(const torch::Tensor & A, const torch::Tensor & A_grad)
    {
        auto G_body = torch::linalg::solve(A.detach(), A_grad);

        auto phi = G_body.index({"...", Slice(None, 3), 3});
        auto R_block = G_body.index({"...", Slice(None, 3), Slice(None, 3)});
        auto S = R_block - R_block.transpose(-1, -2);

        auto rho =
            0.5
            * torch::stack(
                {S.index({"...", 2, 1}), S.index({"...", 0, 2}), S.index({"...", 1, 0})}, -1);

        return torch::cat({rho, phi}, -1);
    }

    torch::Tensor torchSe3ExpMap(const torch::Tensor & xi, double step_r = 1.0, double step_t = 1.0)
    {
        auto rho = xi.index({"...", Slice(None, 3)}) * step_r;
        auto phi = xi.index({"...", Slice(3, None)}) * step_t;
        auto theta = torch::norm(rho, 2, -1);

        auto eye3 = torch::eye(3, torch::TensorOptions().device(xi.device()).dtype(xi.dtype()));
        auto eye4 = torch::eye(4, torch::TensorOptions().device(xi.device()).dtype(xi.dtype()));

        auto small = theta < 1e-18;
        auto skew_rho = skewSymmetric(rho);

        // --- Small angle case ---
        auto skew_rho_sq = torch::matmul(skew_rho, skew_rho);
        auto R_small = eye3 + skew_rho + 0.5 * skew_rho_sq;
        auto V_small = eye3 + 0.5 * skew_rho;
        auto pin_unsqueezed = phi.unsqueeze(-1);
        auto t_small = torch::matmul(V_small, pin_unsqueezed);

        // --- General case ---
        auto axis = rho / theta.clamp_min(1e-18).unsqueeze(-1);
        auto skew_axis = skewSymmetric(axis);

        auto sin_theta = torch::sin(theta);
        auto cos_theta = torch::cos(theta);

        // Broadcasting helper: need to reshape scalars to (..., 1, 1) for
        // matrix broadcasting
        auto sin_u = sin_theta.unsqueeze(-1).unsqueeze(-1);
        auto cos_u = cos_theta.unsqueeze(-1).unsqueeze(-1);
        auto one_minus_cos_u = (1 - cos_theta).unsqueeze(-1).unsqueeze(-1);
        // axis[..., :, None] @ axis[..., None, :] (Outer product)
        auto axis_outer = torch::matmul(axis.unsqueeze(-1), axis.unsqueeze(-2));
        auto R_general = cos_u * eye3 + one_minus_cos_u * axis_outer + sin_u * skew_axis;

        // coef1 = (1 - cos_theta) / (theta**2)
        auto coef1 = (1 - cos_theta) / theta.pow(2);
        // coef2 = (theta - sin_theta) / (theta**3)
        auto coef2 = (theta - sin_theta) / theta.pow(3);
        auto V = eye3 + coef1.unsqueeze(-1).unsqueeze(-1) * skew_rho
                 + coef2.unsqueeze(-1).unsqueeze(-1) * skew_rho_sq;
        auto t_general = torch::matmul(V, pin_unsqueezed);

        // --- Choose by mask ---
        auto small_expanded = small.unsqueeze(-1).unsqueeze(-1);
        auto R = torch::where(small_expanded, R_small, R_general);
        auto t = torch::where(small_expanded, t_small, t_general).index({"...", 0});

        // --- Construct T ---
        std::vector<int64_t> batch_shape(xi.sizes().begin(), xi.sizes().end() - 1);
        std::vector<int64_t> target_shape = batch_shape;
        target_shape.push_back(4);
        target_shape.push_back(4);

        auto T = eye4.expand(target_shape).clone();
        T.index_put_({"...", Slice(None, 3), Slice(None, 3)}, R);
        T.index_put_({"...", Slice(None, 3), 3}, t);

        return T;
    }

    torch::Tensor torchSo3LogMap(const torch::Tensor & R)
    {
        auto trace_R = R.index({"...", 0, 0}) + R.index({"...", 1, 1}) + R.index({"...", 2, 2});
        auto theta = torch::acos(torch::clamp((trace_R - 1) / 2.0, -1.0, 1.0));

        auto skew_sym = 0.5 * (R - R.transpose(-1, -2));
        auto small = theta < 1e-18;

        auto rho_small = torch::stack(
            {skew_sym.index({"...", 2, 1}), skew_sym.index({"...", 0, 2}),
             skew_sym.index({"...", 1, 0})},
            -1);

        auto skew_sym_normed = skew_sym / torch::sin(theta).unsqueeze(-1).unsqueeze(-1);
        auto w = torch::stack(
            {skew_sym_normed.index({"...", 2, 1}), skew_sym_normed.index({"...", 0, 2}),
             skew_sym_normed.index({"...", 1, 0})},
            -1);
        auto rho_general = theta.unsqueeze(-1) * w;

        return torch::where(small.unsqueeze(-1), rho_small, rho_general);
    }

    torch::Tensor torchSe3LogMap(const torch::Tensor & T, double step_r = 1.0, double step_t = 1.0)
    {
        auto R = T.index({"...", Slice(None, 3), Slice(None, 3)});
        auto t = T.index({"...", Slice(None, 3), 3});
        auto rho = torchSo3LogMap(R);
        auto theta = torch::norm(rho, 2, -1);

        auto skew_rho = skewSymmetric(rho);
        auto small = theta < 1e-18;

        auto eye3 = torch::eye(3, torch::TensorOptions().device(T.device()).dtype(T.dtype()));

        // --- Small angle case ---
        auto V_inv_small = eye3 - 0.5 * skew_rho;
        auto phi_small = torch::matmul(V_inv_small, t.unsqueeze(-1)).index({"...", 0});

        // --- General case ---
        auto half_theta = theta / 2.0;
        auto cot_half_theta = torch::cos(half_theta) / torch::sin(half_theta);
        auto coef = (1.0 / theta.pow(2)) * (1.0 - half_theta * cot_half_theta);

        auto V_inv = eye3 - 0.5 * skew_rho
                     + coef.unsqueeze(-1).unsqueeze(-1) * torch::matmul(skew_rho, skew_rho);
        auto phi_general = torch::matmul(V_inv, t.unsqueeze(-1)).index({"...", 0});
        auto phi = torch::where(small.unsqueeze(-1), phi_small, phi_general);
        auto xi = torch::cat({rho / step_r, phi / step_t}, -1);

        return xi;
    }

} // namespace diffcoal

#endif // __diffcoal_utils_helpers_hpp__
