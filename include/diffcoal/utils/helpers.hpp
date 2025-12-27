#ifndef __diffcoal_utils_helpers_hpp__
#define __diffcoal_utils_helpers_hpp__

#include <torch/torch.h>
#include <open3d/Open3D.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>

#include "diffcoal/utils/logger.hpp"

namespace diffcoal
{
    using namespace torch::indexing;

    // Utility class to standardize tensor device and dtype conversion.
    struct DCTensorSpec
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
        auto skew_rho = skewSymmetric(rho);

        auto G_body = torch::zeros_like(A);
        G_body.index_put_({"...", Slice(None, 3), Slice(None, 3)}, skew_rho);
        G_body.index_put_({"...", Slice(None, 3), 3}, phi);

        return torch::matmul(A, G_body);
    }

    torch::Tensor torchMatrixGradToSe3(const torch::Tensor & A, const torch::Tensor & A_grad)
    {
        auto G_body = torch::linalg::solve(A.detach(), A_grad, true);

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

    torch::Tensor adjointFromTransform(const torch::Tensor & T)
    {
        TORCH_CHECK(
            T.size(-1) == 4 && T.size(-2) == 4, "Input must be (...,4,4) transform matrices");

        auto R = T.index({"...", Slice(None, 3), Slice(None, 3)});
        auto p = T.index({"...", Slice(None, 3), 3});

        auto px = skewSymmetric(p);

        // Build Adjoint matrix
        std::vector<int64_t> ad_shape(T.sizes().begin(), T.sizes().end() - 2);
        ad_shape.push_back(6);
        ad_shape.push_back(6);

        auto Ad =
            torch::zeros(ad_shape, torch::TensorOptions().device(T.device()).dtype(T.dtype()));

        Ad.index_put_({"...", Slice(None, 3), Slice(None, 3)}, R);
        Ad.index_put_({"...", Slice(3, None), Slice(None, 3)}, torch::matmul(px, R));
        Ad.index_put_({"...", Slice(3, None), Slice(3, None)}, R);

        return Ad;
    }

    torch::Tensor eqvGrad(
        const torch::Tensor & T1,
        const torch::Tensor & T2,
        const torch::Tensor & grad_T1,
        double step_r = 1.0,
        double step_t = 1.0)
    {
        auto se3_T1 = torchMatrixGradToSe3(T1, grad_T1);
        se3_T1.index_put_({"...", Slice(None, 3)}, se3_T1.index({"...", Slice(None, 3)}) * step_r);
        se3_T1.index_put_({"...", Slice(3, None)}, se3_T1.index({"...", Slice(3, None)}) * step_t);

        auto T_rel = torch::linalg::solve(T2, T1, true);

        auto se3_T2 = torch::einsum("...ij, ...j-> ...i", {adjointFromTransform(T_rel), se3_T1});

        se3_T2.index_put_({"...", Slice(None, 3)}, se3_T2.index({"...", Slice(None, 3)}) / step_r);
        se3_T2.index_put_({"...", Slice(3, None)}, se3_T2.index({"...", Slice(3, None)}) / step_t);

        auto grad_T2 = torchSe3ToMatrixGrad(T2, -se3_T2);

        return grad_T2;
    }

    static bool _local_sample_warn_once = false;

    torch::Tensor localSampleWDthre(
        torch::Tensor global_sample,
        torch::Tensor target_point,
        double dist_thre,
        double min_thre,
        int64_t n_local,
        const std::string & sample_strategy)
    {
        auto dist = torch::norm(
            global_sample - global_sample.index({Slice(), Slice(-1, None)}), 2, -1); // (B, S)

        torch::Tensor valid;
        auto min_thre_tensor = torch::tensor(min_thre, global_sample.options());
        auto dist_thre_tensor = torch::tensor(dist_thre, global_sample.options());

        if (sample_strategy == "adp")
        {
            auto dist2 =
                torch::norm(global_sample.index({Slice(), -1}) - target_point, 2, -1); // (B,)

            valid = dist < torch::maximum(2 * dist2, min_thre_tensor).unsqueeze(-1); // (B, S)
        }
        else if (sample_strategy == "fix")
        {
            valid =
                dist < torch::maximum(dist_thre_tensor, min_thre_tensor).unsqueeze(-1); // (B, S)
        }
        else
        {
            throw std::invalid_argument(
                "Unknown sample strategy " + sample_strategy + ". Available: 'adp', 'fix'.");
        }

        // If buggy batches found, re-sample with adjusted threshold
        auto valid_sum = valid.sum(-1);
        int64_t threshold_count = std::max(static_cast<int>(n_local / 4), 2);
        auto buggy_idx = torch::where(valid_sum < threshold_count)[0];
        if (buggy_idx.numel() > 0)
        {
            std::cerr << diffcoal::logging::WARNING << "Found " << buggy_idx.numel()
                      << " batches (out of " << valid.size(0)
                      << ") with insufficient local samples." << std::endl;

            if (!_local_sample_warn_once)
            {
                std::cerr << diffcoal::logging::WARNING
                          << "If this appears frequently across meshes, consider "
                             "increasing `dthre` or `n_global` in the config. "
                          << "If only for specific meshes with very few batches, those meshes may "
                             "be problematic (e.g., containing disconnected pieces)."
                          << std::endl;
                _local_sample_warn_once = true;
            }

            auto dist_buggy = dist.index({buggy_idx});
            auto topk_result = torch::topk(dist.index({buggy_idx}), n_local, -1, /*largest=*/false);
            auto new_thre = std::get<0>(topk_result).index({Slice(), -1});

            valid.index_put_({buggy_idx}, dist_buggy < new_thre.unsqueeze(-1));
        }

        auto probs = valid / valid.sum(-1, /*keepdim=*/true);
        // NOTE: replacement=False may get samples with prob=0
        auto sampled_idx = torch::multinomial(probs, n_local, /*replacement=*/true);
        std::vector<int64_t> expand_shape = {sampled_idx.size(0), sampled_idx.size(1), 3};
        auto sampled_padded = sampled_idx.unsqueeze(-1).expand(expand_shape);

        return global_sample.gather(1, sampled_padded);
    }

    torch::Tensor eigenToTorch(const std::vector<Eigen::Vector3d> & vec)
    {
        if (vec.empty())
            return torch::empty({0, 3});

        auto opts = torch::TensorOptions().dtype(torch::kFloat64);
        return torch::from_blob(
                   const_cast<double *>(vec[0].data()), {static_cast<long>(vec.size()), 3}, opts)
            .to(torch::kFloat32)
            .clone();
    }

    std::pair<torch::Tensor, torch::Tensor> globalSampleVOrF(
        const open3d::geometry::TriangleMesh & coarse_mesh,
        const open3d::geometry::TriangleMesh & fine_mesh,
        int n_sample,
        const std::string & type_sample)
    {
        if (type_sample == "v")
        {
            int64_t num_vertices = fine_mesh.vertices_.size();
            auto v_ind = torch::randint(0, num_vertices, {n_sample}, torch::kLong);

            auto all_vertices = eigenToTorch(fine_mesh.vertices_);
            auto all_normals = eigenToTorch(fine_mesh.vertex_normals_);

            if (all_normals.size(0) == 0)
            {
                std::cerr << diffcoal::logging::WARNING << "Fine mesh has no vertex normals!"
                          << std::endl;
                // TODO: fineMesh.ComputeVertexNormals();
            }

            auto p = all_vertices.index_select(0, v_ind);
            auto n = all_normals.index_select(0, v_ind);

            return {p, n};
        }
        else if (type_sample == "f")
        {
            // --- Face/Surface Sampling ---
            auto pcd = coarse_mesh_t.SamplePointsPoissonDisk(n_sample);
            auto query_points = eigenToTorch(pcd->points_); // (N, 3)
            auto fine_mesh_T = open3d::t::geometry::TriangleMesh::FromLegacy(fine_mesh);

            open3d::t::geometry::RaycastingScene scene;
            scene.AddTriangles(fine_mesh_T);

            // 执行最近点查询
            open3d::core::Tensor query_points_o3d = open3d::core::Tensor::Init(
                query_points.data_ptr<float>(), {n_sample, 3}, open3d::core::Dtype::Float32);
            auto ans = scene.ComputeClosestPoint(query_points_o3d);

            // 获取结果
            // 'points': 投影后的点坐标 (N, 3)
            // 'primitive_ids': 最近的三角形索引 (N,)
            auto p_tensor_o3d = ans["points"];
            auto f_ids_o3d = ans["primitive_ids"];

            // 转回 LibTorch Tensor
            // Open3D Tensor -> Blob -> LibTorch Tensor (Zero copy if possible, otherwise clone)
            auto p =
                torch::from_blob(p_tensor_o3d.GetDataPtr(), {nSample, 3}, torch::kFloat32).clone();
            auto f_indices = torch::from_blob(f_ids_o3d.GetDataPtr(), {nSample}, torch::kInt64)
                                 .clone(); // int64 or int32 depending on Open3D version

            // 获取对应的面法线
            auto all_face_normals = eigenToTorch(fineMesh.triangle_normals_);

            // 确保面法线存在
            if (all_face_normals.size(0) == 0)
            {
                std::cerr << diffcoal::logging::WARNING << "Warning: Fine mesh has no face normals!"
                          << std::endl;
                // TODO: fineMesh.ComputeTriangleNormals();
            }

            auto n = all_face_normals.index_select(0, f_indices.to(torch::kLong));

            return {p, n};
        }
        else
        {
            throw std::invalid_argument(
                "Unsupported sample type: " + typeSample
                + ". Available choices: 'v' or 'f', indicating vertices and faces. ");
        }
    }

    std::pair<torch::Tensor, torch::Tensor> globalSampleVAndF(
        const open3d::geometry::TriangleMesh & coarse_mesh,
        const open3d::geometry::TriangleMesh & fine_mesh,
        int n_sample)
    {
        auto [vp, vn] = globalSampleVOrF(coarse_mesh, fineMesh, n_sample / 2, "v");
        auto [sp, sn] = globalSampleVOrF(fine_mesh, fineMesh, n_sample / 2, "f");

        return {torch::cat({vp, sp}, 0), torch::cat({vn, sn}, 0)};
    }

    /**
     * Batched shortest distance from points to triangles in 3D.
     * Also returns the closest points on the triangles.
     *
     * Args:
     *     points: (N, 3) tensor
     *     triangles: (N, 3, 3) tensor
     *     eps: small value to avoid division by zero
     *
     * Returns:
     *     std::pair containing:
     *     - distances: (N,) tensor
     *     - closest_points: (N, 3) tensor
     */
    std::pair<torch::Tensor, torch::Tensor> pointToTriangleDistanceAndClosest(
        const torch::Tensor & points, const torch::Tensor & triangles, double eps = 1e-20)
    {

        // 1. Extract vertices (N, 3)
        auto verts = triangles.unbind(1);
        auto A = verts[0];
        auto B = verts[1];
        auto C = verts[2];

        // 2. Compute edges
        auto AB = B - A;
        auto AC = C - A;
        auto BC = C - B;

        // Vector from A to point
        auto AP = points - A;

        // 3. Compute barycentric coordinates
        auto dot00 = (AB * AB).sum(1);
        auto dot01 = (AB * AC).sum(1);
        auto dot02 = (AB * AP).sum(1);
        auto dot11 = (AC * AC).sum(1);
        auto dot12 = (AC * AP).sum(1);

        auto inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + eps);
        auto u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        auto v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        // 4. Check if inside triangle
        auto inside = (u >= 0) & (v >= 0) & (u + v <= 1);

        // 5. Projection onto plane: A + u*AB + v*AC
        auto proj = A + u.unsqueeze(1) * AB + v.unsqueeze(1) * AC;

        // 6. Closest points on edges
        // Edge AB
        auto t_ab = torch::clamp((AP * AB).sum(1) / (dot00 + eps), 0, 1);
        auto p_ab = A + t_ab.unsqueeze(1) * AB;

        // Edge AC
        auto t_ac = torch::clamp((AP * AC).sum(1) / (dot11 + eps), 0, 1);
        auto p_ac = A + t_ac.unsqueeze(1) * AC;

        // Edge BC
        auto BP = points - B;
        auto dot_bc = (BC * BC).sum(1);
        auto t_bc = torch::clamp((BP * BC).sum(1) / (dot_bc + eps), 0, 1);
        auto p_bc = B + t_bc.unsqueeze(1) * BC;

        // 7. Compute distances
        auto dist_proj = torch::norm(points - proj, 2, 1);
        auto dist_ab = torch::norm(points - p_ab, 2, 1);
        auto dist_ac = torch::norm(points - p_ac, 2, 1);
        auto dist_bc = torch::norm(points - p_bc, 2, 1);

        // 8. Find minimum distance for outside points
        auto min_result = torch::min(torch::stack({dist_ab, dist_ac, dist_bc}, 1), 1);
        auto dist_out = std::get<0>(min_result);
        auto idx_out = std::get<1>(min_result); // (N,) indices [0, 1, or 2]

        // 9. Pick the corresponding closest point for outside
        // Stack points to (N, 3, 3)
        auto stacked_points = torch::stack({p_ab, p_ac, p_bc}, 1);
        auto batch_indices = torch::arange(points.size(0), idx_out.options());
        auto closest_out = stacked_points.index({batch_indices, idx_out});

        // 10. Final results
        // distances: (N,)
        auto distances = torch::where(inside, dist_proj, dist_out);

        // closest_points: (N, 3)
        auto closest_points = torch::where(inside.unsqueeze(1), proj, closest_out);

        return {distances, closest_points};
    }

} // namespace diffcoal

#endif // __diffcoal_utils_helpers_hpp__
