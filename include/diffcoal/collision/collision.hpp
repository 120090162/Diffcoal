#ifndef __diffcoal_collision_collision_hpp__
#define __diffcoal_collision_collision_hpp__

#include <coal/collision_object.h>

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include "diffcoal/collision/fwd.hpp"

namespace diffcoal
{
    /// \brief Load a convex collision geometry from a file and apply scaling.
    ///
    /// This templated helper uses COAL's MeshLoader to load `file_name`, applies
    /// the provided `scale` (length-3 vector) and builds a convex hull.
    ///
    /// 	param Scalar Numeric scalar type for Eigen/COAL interop (e.g., double).
    /// 	param Options Eigen matrix/vector options (row/col major, etc.).
    /// \param file_name Path to mesh file to load.
    /// \param scale Length-3 vector containing per-axis scale factors.
    /// \returns Shared pointer to the constructed convex CollisionGeometry.
    ///
    template<typename Scalar, int Options>
    std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromFile(const std::string & file_name, const std::vector<Scalar> & scale);

    /// \brief Build a convex collision geometry from raw vertex data.
    ///
    /// The input `verts` is expected as a flat row-major array of 3D points:
    /// `[x0,y0,z0, x1,y1,z1, ...]`. The function maps the raw data into an
    /// Eigen matrix and builds a convex hull returned as `CollisionGeometry`.
    ///
    /// 	param Scalar Numeric scalar type for Eigen/COAL interop.
    /// 	param Options Eigen matrix/vector options.
    /// \param verts Flat vector of vertex coordinates (size divisible by 3).
    /// \returns Shared pointer to the constructed convex CollisionGeometry.
    ///
    template<typename Scalar, int Options>
    std::shared_ptr<const coal::CollisionGeometry>
    getConvexFromData(const std::vector<Scalar> & verts);

    /// \brief Compute batched distances between many convex/mesh pairs in parallel.
    ///
    /// This templated routine evaluates distances for a set of selected convex
    /// pairs (described by index lists and flattened pose arrays). Results are
    /// written into the output vectors passed by reference. The implementation
    /// parallelizes work using OpenMP; callers control the number of threads.
    ///
    /// 	param Scalar Numeric scalar type for results and pose storage.
    /// 	param Options Eigen matrix/vector options.
    /// \param shape_lst Vector of collision geometries used by pair indices.
    /// \param shape1_idx_lst Index of the first shape for each convex pair.
    /// \param pose1_lst Flattened pose arrays for the first object (4x4 per group, row-major).
    /// \param shape2_idx_lst Index of the second shape for each convex pair.
    /// \param pose2_lst Flattened pose arrays for the second object (4x4 per group, row-major).
    /// \param group_idx_lst Mapping from convex-pair index to mesh-group index.
    /// \param valid_idx_lst List of selected pair indices to evaluate (length n_valid).
    /// \param n_batch Number of batches.
    /// \param n_pair Number of convex piece pairs per batch.
    /// \param n_group Number of mesh pairs per batch.
    /// \param n_valid Number of selected pairs to process.
    /// \param n_thread Requested number of OpenMP threads.
    /// \param dist_result Output vector (size total_groups) to receive minimum distances.
    /// \param normal_result Output vector (size total_groups*3) to receive normals.
    /// \param wp1_result Output vector (size total_groups*3) to receive nearest point on shape1.
    /// \param wp2_result Output vector (size total_groups*3) to receive nearest point on shape2.
    /// \param min_idx_result Output vector (size total_groups) to receive index of best convex
    /// pair.
    /// \returns void (results are written into output parameters).
    ///
    template<typename Scalar, int Options>
    void batchedCoalDistance(
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
        std::vector<size_t> & min_idx_result);

    /// \brief Gather neighbor vertices for convex shapes given separation vectors.
    ///
    /// For each selected convex shape, the function finds vertices near the
    /// direction `sep_vec` using BFS on the convex adjacency graph, shuffles
    /// candidates, and writes `n_nbr` neighbor points into `neighbor_result`.
    ///
    /// 	param Scalar Numeric scalar type used for separation vectors/results.
    /// 	param Options Eigen matrix/vector options.
    /// \param shape_lst Vector of collision geometries.
    /// \param valid_idx_lst List of selected convex indices (length n_valid).
    /// \param sep_vec_lst Flattened separation vectors (length n_valid*3).
    /// \param n_valid Number of valid selections.
    /// \param n_level BFS expansion depth (levels).
    /// \param n_nbr Number of neighbor points to output per selection.
    /// \param n_thread Requested number of OpenMP threads.
    /// \param neighbor_result Output vector (size n_valid * n_nbr * 3) filled with neighbor points.
    /// \returns void (results are written into `neighbor_result`).
    ///
    template<typename Scalar, int Options>
    void batchedGetNeighbor(
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> &
            shape_lst, // pass by const ref
        const std::vector<size_t> & valid_idx_lst,
        const std::vector<Scalar> & sep_vec_lst, // assumed shape (n_valid, 3)
        const size_t n_valid,
        const size_t n_level,
        const size_t n_nbr,
        const int n_thread,
        std::vector<Scalar> & neighbor_result);
} // namespace diffcoal

/* --- Details -------------------------------------------------------------------- */
#include "diffcoal/collision/collision.hxx"

#endif // ifndef __diffcoal_collision_collision_hpp__