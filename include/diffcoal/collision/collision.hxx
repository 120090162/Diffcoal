#ifndef __diffcoal_collision_collision_hxx__
#define __diffcoal_collision_collision_hxx__

#include <coal/distance.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>
#include <coal/BVH/BVH_model.h>
#include <coal/mesh_loader/loader.h>

#include "diffcoal/utils/openmp.hpp"

#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <random>

namespace diffcoal
{
    // \brief Load a convex collision geometry from a file and apply scaling.
    //
    // Uses COAL's MeshLoader to read `file_name`, applies the provided
    // per-axis `scale`, builds a convex hull and returns the resulting
    // CollisionGeometry. See getConvexFromFile declaration for details.
    template<typename Scalar, int Options>
    inline std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromFile(const std::string & file_name, const std::vector<Scalar> & scale)
    {
        typedef Eigen::Matrix<Scalar, 3, 1, Options> Vector3;

        coal::NODE_TYPE bv_type = coal::BV_AABB;
        coal::MeshLoader loader(bv_type);
        const Scalar * scale_ptr = scale.data();
        Vector3 eigen_scale(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
        coal::BVHModelPtr_t bvh = loader.load(file_name, eigen_scale);
        bvh->buildConvexHull(true, "Qt");
        return bvh->convex;
    }

    // \brief Build a convex geometry from flat vertex data.
    //
    // The input `verts` is a flat row-major array of 3D points and will be
    // mapped into an Eigen matrix with shape (n_rows x 3) before being
    // passed to COAL to construct the convex hull.
    template<typename Scalar, int Options>
    inline std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromData(const std::vector<Scalar> & verts)
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor> MatrixX3;

        size_t n_rows = verts.size() / 3;

        Eigen::Map<const MatrixX3> eigen_verts(verts.data(), n_rows, 3);
        auto bvh = std::make_shared<coal::BVHModel<coal::AABB>>();
        bvh->beginModel();
        bvh->addVertices(eigen_verts);
        bvh->endModel();
        bvh->buildConvexHull(true, "Qt");
        return bvh->convex;
    }

    // --------------------------------------------------------------------------
    template<typename Scalar, int Options>
    inline void batchedCoalDistance(
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> &
            shape_lst, // Use CollisionGeometry here
        const std::vector<size_t> & shape1_idx_lst,
        const std::vector<Scalar> & pose1_lst,
        const std::vector<size_t> & shape2_idx_lst,
        const std::vector<Scalar> & pose2_lst,
        const std::vector<size_t> & group_idx_lst,
        const std::vector<size_t> & valid_idx_lst,
        const size_t n_batch,
        const size_t n_pair,  // convex piece pairs
        const size_t n_group, // mesh pairs
        const size_t n_valid,
        const int n_thread,
        std::vector<Scalar> & dist_result,
        std::vector<Scalar> & normal_result,
        std::vector<Scalar> & wp1_result,
        std::vector<Scalar> & wp2_result,
        std::vector<size_t> & min_idx_result)
    {
        typedef Eigen::Matrix<Scalar, 3, 1, Options> Vector3;
        typedef Eigen::Matrix<Scalar, 3, 3, Options> Matrix3;

        const Scalar * pose1_ptr = pose1_lst.data();
        const size_t * shape1_idx_ptr = shape1_idx_lst.data();
        const Scalar * pose2_ptr = pose2_lst.data();
        const size_t * shape2_idx_ptr = shape2_idx_lst.data();
        const size_t * select_idx_ptr = valid_idx_lst.data();
        const size_t * group_idx_ptr = group_idx_lst.data();

        Scalar * dist_result_ptr = dist_result.data();
        Scalar * normal_result_ptr = normal_result.data();
        Scalar * wp1_result_ptr = wp1_result.data();
        Scalar * wp2_result_ptr = wp2_result.data();
        size_t * min_idx_result_ptr = min_idx_result.data();

        size_t total_groups = n_batch * n_group;

        setDefaultOpenMPSettings(n_thread);

        // Check the real number of threads
        int T = 0;
#pragma omp parallel
        {
#pragma omp single
            T = omp_get_num_threads();
        }

        // Allocate and initialize
        std::vector<size_t> local_idx(T * total_groups, 0);
        std::vector<coal::DistanceResult> local_res;
        local_res.resize((size_t)T * total_groups);
        coal::DistanceResult inf_dr;
        inf_dr.min_distance = std::numeric_limits<Scalar>::infinity();
        for (size_t i = 0; i < local_res.size(); ++i)
        {
            local_res[i] = inf_dr;
        }

// Main parallel loop: compute distances and update thread-local per-group minima
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t * my_local_idx = &local_idx[(size_t)tid * total_groups];
            coal::DistanceResult * my_local_res = &local_res[(size_t)tid * total_groups];

#pragma omp for schedule(static)
            for (size_t i = 0; i < n_valid; ++i)
            {
                size_t idx = select_idx_ptr[i];
                size_t pair_idx = idx % n_pair;
                size_t batch_idx = idx / n_pair;
                size_t group_idx = batch_idx * n_group + group_idx_ptr[pair_idx];

                // Build transforms
                Matrix3 R1;
                R1 << pose1_ptr[16 * group_idx], pose1_ptr[16 * group_idx + 1],
                    pose1_ptr[16 * group_idx + 2], pose1_ptr[16 * group_idx + 4],
                    pose1_ptr[16 * group_idx + 5], pose1_ptr[16 * group_idx + 6],
                    pose1_ptr[16 * group_idx + 8], pose1_ptr[16 * group_idx + 9],
                    pose1_ptr[16 * group_idx + 10];
                Vector3 T1(
                    pose1_ptr[16 * group_idx + 3], pose1_ptr[16 * group_idx + 7],
                    pose1_ptr[16 * group_idx + 11]);
                coal::Transform3s transform1(R1, T1);

                Matrix3 R2;
                R2 << pose2_ptr[16 * group_idx], pose2_ptr[16 * group_idx + 1],
                    pose2_ptr[16 * group_idx + 2], pose2_ptr[16 * group_idx + 4],
                    pose2_ptr[16 * group_idx + 5], pose2_ptr[16 * group_idx + 6],
                    pose2_ptr[16 * group_idx + 8], pose2_ptr[16 * group_idx + 9],
                    pose2_ptr[16 * group_idx + 10];
                Vector3 T2(
                    pose2_ptr[16 * group_idx + 3], pose2_ptr[16 * group_idx + 7],
                    pose2_ptr[16 * group_idx + 11]);
                coal::Transform3s transform2(R2, T2);

                // Call COAL distance (CPU)
                coal::DistanceRequest dist_req;
                coal::DistanceResult dist_res;
                coal::distance(
                    shape_lst[shape1_idx_ptr[pair_idx]].get(), transform1,
                    shape_lst[shape2_idx_ptr[pair_idx]].get(), transform2, dist_req, dist_res);

                // Update thread-local best
                if (dist_res.min_distance < my_local_res[group_idx].min_distance)
                {
                    my_local_res[group_idx] = dist_res;
                    my_local_idx[group_idx] = pair_idx;
                }
            } // end for
        } // end parallel

// Parallel reduction across threads per group -> write to output arrays
#pragma omp parallel for schedule(static)
        for (size_t g = 0; g < total_groups; ++g)
        {
            size_t best_idx = 0;
            coal::DistanceResult best = inf_dr;
            for (int t = 0; t < T; ++t)
            {
                const coal::DistanceResult & cand = local_res[(size_t)t * total_groups + g];
                if (cand.min_distance < best.min_distance)
                {
                    best = cand;
                    best_idx = local_idx[(size_t)t * total_groups + g];
                }
            }
            // Write outputs (if no sample, min_distance remains +inf and points are zero)
            dist_result_ptr[g] = best.min_distance;
            normal_result_ptr[3 * g + 0] = best.normal[0];
            normal_result_ptr[3 * g + 1] = best.normal[1];
            normal_result_ptr[3 * g + 2] = best.normal[2];
            wp1_result_ptr[3 * g + 0] = best.nearest_points[0][0];
            wp1_result_ptr[3 * g + 1] = best.nearest_points[0][1];
            wp1_result_ptr[3 * g + 2] = best.nearest_points[0][2];
            wp2_result_ptr[3 * g + 0] = best.nearest_points[1][0];
            wp2_result_ptr[3 * g + 1] = best.nearest_points[1][1];
            wp2_result_ptr[3 * g + 2] = best.nearest_points[1][2];
            min_idx_result_ptr[g] = best_idx;
        }
    }

    // --------------------------------------------------------------------------
    template<typename Scalar, int Options>
    inline void batchedGetNeighbor(
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> &
            shape_lst, // pass by const ref
        const std::vector<size_t> & valid_idx_lst,
        const std::vector<Scalar> & sep_vec_lst, // assumed shape (n_valid, 3)
        const size_t n_valid,
        const size_t n_level,
        const size_t n_nbr,
        const int n_thread,
        std::vector<Scalar> & neighbor_result)
    {
        typedef Eigen::Matrix<Scalar, 3, 1, Options> Vector3;

        const Scalar * sep_vec_ptr = sep_vec_lst.data();
        const size_t * select_idx_ptr = valid_idx_lst.data();
        Scalar * neighbor_result_ptr = neighbor_result.data();

        setDefaultOpenMPSettings(n_thread);

#pragma omp parallel for
        for (size_t i = 0; i < n_valid; ++i)
        {
            const size_t sel_idx = select_idx_ptr[i];
            const std::shared_ptr<const coal::CollisionGeometry> & geom_ptr = shape_lst[sel_idx];
            const coal::ConvexBase * convex_base =
                dynamic_cast<const coal::ConvexBase *>(geom_ptr.get());

            Vector3 sep_vec;
            sep_vec(0) = sep_vec_ptr[3 * i + 0];
            sep_vec(1) = sep_vec_ptr[3 * i + 1];
            sep_vec(2) = sep_vec_ptr[3 * i + 2];

            const std::vector<coal::Vec3s> & pts = *(convex_base->points);

            // find vertex with maximum dot
            Scalar maxdot = -std::numeric_limits<Scalar>::infinity();
            size_t max_id = 0;
            for (size_t j = 0; j < pts.size(); ++j)
            {
                const Scalar dot = pts[j].dot(sep_vec);
                if (dot > maxdot)
                {
                    maxdot = dot;
                    max_id = j;
                }
            }
            std::vector<size_t> neighbor_lst;
            std::vector<size_t> level_lst;
            neighbor_lst.push_back(max_id);
            level_lst.push_back(0);

            // BFS
            size_t curr_idx = 0;
            const std::vector<coal::ConvexBase::Neighbors> & neighbors = *(convex_base->neighbors);
            while (curr_idx < neighbor_lst.size())
            {
                size_t vertex_idx = neighbor_lst[curr_idx];
                size_t curr_level = level_lst[curr_idx];
                if (curr_level < n_level)
                {
                    const coal::ConvexBase::Neighbors & point_neighbors = neighbors[vertex_idx];
                    const size_t cnt = static_cast<size_t>(point_neighbors.count());
                    for (size_t jj = 0; jj < cnt; ++jj)
                    {
                        size_t neighbor_index = point_neighbors[static_cast<int>(jj)];
                        auto it =
                            std::find(neighbor_lst.begin(), neighbor_lst.end(), neighbor_index);
                        if (it == neighbor_lst.end())
                        {
                            level_lst.push_back(curr_level + 1);
                            neighbor_lst.push_back(neighbor_index);
                        }
                    }
                }
                ++curr_idx;
            }

            // per-iteration random generator (seed depends on i to avoid identical shuffles across
            // threads)
            std::mt19937 gen(static_cast<size_t>(i + 123456));
            std::shuffle(neighbor_lst.begin(), neighbor_lst.end(), gen);

            // fill results
            for (size_t j = 0; j < n_nbr; ++j)
            {
                size_t idx = neighbor_lst[j % neighbor_lst.size()];
                neighbor_result_ptr[3 * (n_nbr * i + j) + 0] = pts[idx][0];
                neighbor_result_ptr[3 * (n_nbr * i + j) + 1] = pts[idx][1];
                neighbor_result_ptr[3 * (n_nbr * i + j) + 2] = pts[idx][2];
            }
        } // end omp for
    }

} // namespace diffcoal

#endif // ifndef __diffcoal_collision_collision_hxx__