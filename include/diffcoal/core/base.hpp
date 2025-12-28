#ifndef __diffcoal_core_base_hpp__
#define __diffcoal_core_base_hpp__

#include <torch/torch.h>

#include "diffcoal/core/fwd.hpp"
#include "diffcoal/collision/collision.hpp"
#include "diffcoal/utils/helpers.hpp"
#include "diffcoal/utils/mesh_io.hpp"

namespace diffcoal
{
    template<typename T>
    std::vector<T> tensor_to_vector(const torch::Tensor & t)
    {
        // 确定目标 PyTorch 类型
        torch::ScalarType target_dtype;
        if constexpr (std::is_same_v<T, double>)
            target_dtype = torch::kDouble;
        else if constexpr (std::is_same_v<T, float>)
            target_dtype = torch::kFloat;
        else if constexpr (std::is_same_v<T, size_t>)
            target_dtype = torch::kLong; // size_t 用 int64
        else
            target_dtype = t.scalar_type();

        // 1. 转到 CPU
        // 2. 转换为目标数据类型 (例如 float -> double)
        // 3. 确保内存连续
        auto t_cpu = t.to(torch::kCPU).to(target_dtype).contiguous();

        std::vector<T> vec(t_cpu.numel());

        // 处理数据复制
        if constexpr (std::is_same_v<T, size_t> && sizeof(size_t) == sizeof(int64_t))
        {
            // 如果 size_t 和 int64_t 大小一致，直接内存拷贝
            std::memcpy(vec.data(), t_cpu.data_ptr(), t_cpu.numel() * sizeof(T));
        }
        else
        {
            // 通用复制 (处理 float->double, 或 size_t!=int64_t)
            // 获取对应 Tensor 类型的指针
            using TensorType =
                typename std::conditional<std::is_same_v<T, size_t>, int64_t, T>::type;
            const TensorType * ptr = t_cpu.data_ptr<TensorType>();
            for (int64_t i = 0; i < t_cpu.numel(); ++i)
            {
                vec[i] = static_cast<T>(ptr[i]);
            }
        }
        return vec;
    }

    torch::Tensor vector_double_to_tensor(
        const std::vector<double> & vec,
        const std::vector<int64_t> & shape,
        const std::shared_ptr<DCTensorSpec> & ts)
    {
        // 1. 从 vector 创建 float64 tensor
        auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
        // 注意：这里需要拷贝，因为 vector 稍后会被销毁
        auto t_double = torch::tensor(vec, options).reshape(shape);

        // 2. 转换回目标设备和类型 (ts->to 负责做这件事)
        return ts->to(t_double);
    }

    class BaseConfig
    {
    public:
        // --- Public API ---
        int n_thread = 16; // cpu thread number for coal library

        // for adaptive sampling and visualization
        torch::Tensor tp1_o; // Optional
        torch::Tensor tp2_o; // Optional

        bool egt = true; // whether to enable equivalent gradient transport

        // float type is common for torch tensors
        float egt_step_r = 1.0;   // the relative step between r and t matters
        float egt_step_t = 0.001; // the relative step between r and t matters

        // --- Internal Fields ---
        std::vector<std::shared_ptr<DCMesh>> _meshes;
        torch::Tensor _collision_pairs; // (N_pair, 2)
        std::vector<std::shared_ptr<const coal::CollisionGeometry>> _cvx_lst;
        torch::Tensor _sph_lst;
        std::shared_ptr<DCTensorSpec> _ts;

        torch::Tensor _cvx_n_sum;
        torch::Tensor _cvx_min_idx; // convex piece id that the witness point lies on

        // Index mappings
        torch::Tensor _ml2mp_idx1, _ml2mp_idx2; // mesh list -> mesh pair
        torch::Tensor _cl2cp_idx1, _cl2cp_idx2; // convex piece list -> convex piece pair
        torch::Tensor _mp2cp_idx1, _mp2cp_idx2; // mesh pair -> convex piece pair
        torch::Tensor _cp2mp_idx;               // convex piece pair -> mesh pair

        BaseConfig(
            std::vector<std::shared_ptr<DCMesh>> meshes,
            torch::Tensor collision_pairs = torch::Tensor())
        : _meshes(meshes)
        {
            // 1. Init TensorSpec & WarpSphereDist
            if (_meshes.empty())
                throw std::runtime_error("Meshes cannot be empty");

            auto & m0 = *_meshes[0];
            _ts = std::make_shared<DCTensorSpec>(
                m0.getBoundingSpheres().device(), m0.getBoundingSpheres().scalar_type());

            int64_t n_mesh = _meshes.size();

            // 2. Init Collision Pairs
            if (!collision_pairs.defined())
            {
                // torch.triu_indices(n_mesh, n_mesh, offset=1).T
                auto indices = torch::triu_indices(n_mesh, n_mesh, 1);
                _collision_pairs = indices.t().contiguous();
            }
            else
            {
                _collision_pairs = collision_pairs;
            }

            // 3. Aggregate Spheres and Convex Pieces
            std::vector<torch::Tensor> sph_vec;
            std::vector<int64_t> cvx_n_sum_vec;
            int64_t n_cvx_sum = 0;

            _cvx_lst.clear();
            for (auto & m : _meshes)
            {
                sph_vec.push_back(m->getBoundingSpheres());
                _cvx_lst.insert(
                    _cvx_lst.end(), m->getConvexPieces().begin(), m->getConvexPieces().end());
                cvx_n_sum_vec.push_back(n_cvx_sum);
                n_cvx_sum += m->getNCvx();
            }

            _sph_lst = torch::cat(sph_vec, 0);
            _cvx_n_sum = torch::tensor(
                cvx_n_sum_vec, torch::dtype(torch::kLong)
                                   .device(torch::kCPU)); // Keep on CPU for list indexing usually

            // 4. Update internal indices
            update_collision_pairs(_collision_pairs, tp1_o, tp2_o);
        }

        void update_collision_pairs(
            torch::Tensor collision_pairs, torch::Tensor tp1_in, torch::Tensor tp2_in)
        {
            _collision_pairs = collision_pairs;
            tp1_o = tp1_in;
            tp2_o = tp2_in;

            std::vector<torch::Tensor> cp2mp_vec;
            std::vector<torch::Tensor> cl2cp1_vec, cl2cp2_vec;
            std::vector<torch::Tensor> mp2cp1_vec, mp2cp2_vec;

            // Iterate over pairs
            auto pairs_cpu = _collision_pairs.cpu();
            auto pairs_acc = pairs_cpu.accessor<int64_t, 2>();
            int64_t n_pairs = _collision_pairs.size(0);

            for (int64_t i = 0; i < n_pairs; ++i)
            {
                int64_t idx1 = pairs_acc[i][0];
                int64_t idx2 = pairs_acc[i][1];

                int64_t n_cvx1 = _meshes[idx1]->getNCvx();
                int64_t n_cvx2 = _meshes[idx2]->getNCvx();

                // _cp2mp_idx.extend([i] * n_cvx1 * n_cvx2)
                cp2mp_vec.push_back(torch::full({n_cvx1 * n_cvx2}, i, torch::kLong));

                // _cl2cp_idx1
                auto base1 = _cvx_n_sum[idx1].item<int64_t>();
                auto range1 = torch::arange(n_cvx1, torch::kLong);
                cl2cp1_vec.push_back(
                    (base1 + range1).repeat({n_cvx2})); // repeat in PyTorch is repeat in C++

                // _cl2cp_idx2
                auto base2 = _cvx_n_sum[idx2].item<int64_t>();
                auto range2 = torch::arange(n_cvx2, torch::kLong);
                cl2cp2_vec.push_back((base2 + range2).repeat_interleave(n_cvx1));

                // _mp2cp_idx1
                mp2cp1_vec.push_back(torch::full({n_cvx1}, i, torch::kLong).repeat({n_cvx2}));

                // _mp2cp_idx2
                mp2cp2_vec.push_back(
                    torch::full({n_cvx2}, i, torch::kLong).repeat_interleave(n_cvx1));
            }

            // Convert and move to device
            _ml2mp_idx1 = _ts->toIdx(_collision_pairs.select(1, 0));
            _ml2mp_idx2 = _ts->toIdx(_collision_pairs.select(1, 1));

            _cl2cp_idx1 = _ts->toIdx(torch::cat(cl2cp1_vec));
            _cl2cp_idx2 = _ts->toIdx(torch::cat(cl2cp2_vec));
            _mp2cp_idx1 = _ts->toIdx(torch::cat(mp2cp1_vec));
            _mp2cp_idx2 = _ts->toIdx(torch::cat(mp2cp2_vec));
            _cp2mp_idx = _ts->toIdx(torch::cat(cp2mp_vec));
        }
    };

    class BaseCollision : public torch::autograd::Function<BaseCollision>
    {
    public:
        // forward 静态函数
        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext * ctx,
            torch::Tensor T1,
            torch::Tensor T2,
            std::shared_ptr<BaseConfig> cfg,
            void * vis // vis 在 Python 中是对象，这里暂用 void* 或 pybind11::object
        )
        {
            // 解包 config
            auto & cvx_lst = cfg->_cvx_lst;
            auto sph_lst = cfg->_sph_lst;
            auto ts = cfg->_ts;

            int64_t n_batch = T2.size(0);
            int64_t n_mesh_pair = T2.size(1);

            // --- 1. Broad-phase filter (Bounding Spheres) ---

            // idx indexing: sph1_o = sph_lst[cfg._cl2cp_idx1]
            auto sph1_o = sph_lst.index_select(0, cfg->_cl2cp_idx1);
            auto sph2_o = sph_lst.index_select(0, cfg->_cl2cp_idx2);

            // SphereDist forward
            auto [s2s_max, s2s_min] =
                sphereDistForward(T1, T2, sph1_o, sph2_o, cfg->_mp2cp_idx1, cfg->_mp2cp_idx2);

            // batched_pair_idx = cfg._cp2mp_idx[None].expand_as(s2s_max)
            auto batched_pair_idx = cfg->_cp2mp_idx.unsqueeze(0).expand_as(s2s_max);

            // s2s_max_sct init
            int64_t max_cp_idx = cfg->_cp2mp_idx.max().item<int64_t>();
            auto s2s_max_sct = ts->to(torch::zeros({n_batch, max_cp_idx + 1}));

            // s2s_max_sct.scatter_reduce_
            // PyTorch C++ API: scatter_reduce(dim, index, src, reduce="amin", include_self=False)
            // 注意：不同 LibTorch 版本 API 略有差异，这里使用较新版标准写法
            s2s_max_sct.scatter_reduce_(
                1, batched_pair_idx, s2s_max, "amin", /*include_self=*/false);

            // valid = s2s_min - s2s_max_sct.gather(...)
            auto gathered_max = s2s_max_sct.gather(1, batched_pair_idx);
            auto valid = s2s_min - gathered_max;

            // valid_idx = torch.where(valid.view(-1) < 0)[0].cpu().numpy()
            // C++: 获取索引 Tensor 并转为 CPU long 指针
            auto valid_flat = valid.view({-1});
            auto valid_idx_tensor = torch::where(valid_flat < 0)[0].cpu().to(torch::kLong);

            int64_t n_valid = valid_idx_tensor.numel();
            int64_t n_cvx_pair = valid.size(-1);

            // --- 2. Narrow-phase GJK (COAL) ---

            // 1. 将索引 Tensor 转为 vector<size_t>
            auto idx1_vec = tensor_to_vector<size_t>(cfg->_cl2cp_idx1);
            auto idx2_vec = tensor_to_vector<size_t>(cfg->_cl2cp_idx2);
            auto mp2cp_vec = tensor_to_vector<size_t>(cfg->_cp2mp_idx);
            auto valid_idx_vec = tensor_to_vector<size_t>(valid_idx_tensor);

            // 2. 将数据 Tensor 转为 vector<Scalar> (double)
            // 无论 T1/T2 是 float32 还是 float64，都会被安全转换为 double
            auto T1_vec = tensor_to_vector<context::Scalar>(T1);
            auto T2_vec = tensor_to_vector<context::Scalar>(T2);

            // 3. 准备输出 vector<Scalar>
            size_t total_out = n_batch * n_mesh_pair;
            std::vector<context::Scalar> dist_vec(total_out, 100.0);
            std::vector<context::Scalar> normal_vec(total_out * 3, 0.0);
            std::vector<context::Scalar> wp1_vec(total_out * 3, 0.0);
            std::vector<context::Scalar> wp2_vec(total_out * 3, 0.0);
            std::vector<size_t> min_idx_vec(total_out, 0);

            batchedCoalDistance<context::Scalar, context::Options>(
                cvx_lst, idx1_vec, T1_vec, idx2_vec, T2_vec, mp2cp_vec, valid_idx_vec,
                (size_t)n_batch, (size_t)n_cvx_pair, (size_t)n_mesh_pair, n_valid, cfg->n_thread,
                dist_vec, normal_vec, wp1_vec, wp2_vec, min_idx_vec);

            // 5. 将 vector<double> 转换回 Tensor (ts指定的类型，通常是float32)
            auto dist = vector_double_to_tensor(dist_vec, {n_batch, n_mesh_pair}, ts);
            auto normal = vector_double_to_tensor(normal_vec, {n_batch, n_mesh_pair, 3}, ts);
            auto wp1 = vector_double_to_tensor(wp1_vec, {n_batch, n_mesh_pair, 3}, ts);
            auto wp2 = vector_double_to_tensor(wp2_vec, {n_batch, n_mesh_pair, 3}, ts);

            // 处理索引 (size_t -> int64)
            std::vector<int64_t> min_idx_long(min_idx_vec.begin(), min_idx_vec.end());
            auto min_idx_out =
                torch::tensor(min_idx_long, torch::TensorOptions().dtype(torch::kLong))
                    .view({n_batch, n_mesh_pair});

            // d_sign = 2 * (dist > 0) - 1
            auto d_sign = 2 * (dist > 0).to(dist.dtype()) - 1;

            if (dist.max().item<float>() > 1.0f)
            {
                std::cout << "Warning: Distance " << dist.max().item<float>() << std::endl;
            }

            // Update cfg state
            cfg->_cvx_min_idx = ts->toIdx(min_idx_out);

            // Save for backward
            // C++ Autograd 中不能像 Python 那样随意给 ctx 挂属性 (ctx.cfg = cfg)
            // 通常只保存 Tensor。如果需要 cfg，要么将其序列化，要么作为 IValue 保存（较复杂）。
            // 这里的标准做法是只保存 Tensors。
            ctx->save_for_backward({T1, T2, dist, normal, wp1, wp2});

            // Return: wp1, wp2, d_sign
            return {wp1, wp2, d_sign};
        }
    };
} // namespace diffcoal

#endif // ifndef __diffcoal_core_base_hpp__