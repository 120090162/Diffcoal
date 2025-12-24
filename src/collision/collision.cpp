#include "diffcoal/collision/collision.hpp"

namespace diffcoal
{
    template std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromFile<context::Scalar, context::Options>(const std::string & file_name, const std::vector<context::Scalar> & scale);

    template std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromData<context::Scalar, context::Options>(const std::vector<context::Scalar> & verts);

    template void batchedCoalDistance<context::Scalar, context::Options>(
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> & shape_lst, // Use CollisionGeometry here
        const std::vector<size_t> & shape1_idx_lst,
        const std::vector<context::Scalar> & pose1_lst,
        const std::vector<size_t> & shape2_idx_lst,
        const std::vector<context::Scalar> & pose2_lst,
        const std::vector<size_t> & group_idx_lst,
        const std::vector<size_t> & valid_idx_lst,
        const size_t n_batch,
        const size_t n_pair,  // convex piece pairs
        const size_t n_group, // mesh pairs
        const size_t n_valid,
        const int n_thread,
        std::vector<context::Scalar> & dist_result,
        std::vector<context::Scalar> & normal_result,
        std::vector<context::Scalar> & wp1_result,
        std::vector<context::Scalar> & wp2_result,
        std::vector<size_t> & min_idx_result);

    template void batchedGetNeighbor<context::Scalar, context::Options>(
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> & shape_lst, // pass by const ref
        const std::vector<size_t> & valid_idx_lst,
        const std::vector<context::Scalar> & sep_vec_lst, // assumed shape (n_valid, 3)
        const size_t n_valid,
        const size_t n_level,
        const size_t n_nbr,
        const int n_thread,
        std::vector<context::Scalar> & neighbor_result);
} // namespace diffcoal
