#ifndef __diffcoal_utils_mesh_io_hpp__
#define __diffcoal_utils_mesh_io_hpp__

#include <open3d/Open3D.h>
#include <coal/collision_object.h>
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "diffcoal/utils/helpers.hpp"
#include "diffcoal/utils/fwd.hpp"

namespace diffcoal
{
    /**
     * @brief Mesh container for Differentiable Collision detection.
     *
     */
    class DCMesh
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
        /**
         * @brief Construct a DCMesh.
         *
         * @param coarse Shared pointer to the coarse open3d::geometry::TriangleMesh.
         * @param fine Shared pointer to the fine open3d::geometry::TriangleMesh.
         * @param convex_parts Vector of shared_ptr to coal::CollisionGeometry (convex parts).
         * @param spheres Torch tensor of bounding spheres with shape (N,4) (center_x,y,z, radius).
         */
        DCMesh(
            std::shared_ptr<open3d::geometry::TriangleMesh> coarse,
            std::shared_ptr<open3d::geometry::TriangleMesh> fine,
            std::vector<std::shared_ptr<const coal::CollisionGeometry>> convex_parts,
            torch::Tensor spheres);

        /**
         * @brief Create a DCMesh from raw vertex and face arrays.
         *
         * The function will construct a convex hull from the provided mesh data.
         * Intended for simple/convex objects where a single convex piece suffices.
         *
         * @param vertices std::vector<Eigen::Vector3d> (N,3) vertex positions.
         * @param faces std::vector<Eigen::Vector3i> (M,3) triangle indices.
         * @param ts Tensor spec controlling device/dtype for returned tensors.
         * @return DCMesh constructed from the given data.
         */
        static DCMesh fromData(
            const std::vector<Eigen::Vector3d> & vertices,
            const std::vector<Eigen::Vector3i> & faces,
            DCTensorSpec ts = DCTensorSpec());

        /**
         * @brief Load a DCMesh from file or processed directory.
         *
         * Behavior:
         *  - If convex_hull == true:
         *      Accepts either a single .obj file or a directory containing a processed mesh.
         *      Loads the mesh and (optionally) builds a convex hull representation.
         *  - If convex_hull == false:
         *      Expects a MeshProcess-processed directory structure (see MeshProcess repo).
         *
         * @param obj_path Path to .obj file or processed mesh directory.
         * @param scale Uniform scaling factor applied to loaded vertices.
         * @param convex_hull If true, load/convert to convex hull; otherwise expect processed
         * concave data.
         * @param ts Tensor spec controlling device/dtype for returned tensors.
         * @return DCMesh loaded from disk.
         */
        static DCMesh fromFile(
            const std::string & obj_path,
            double scale = 1.0,
            bool convex_hull = true,
            DCTensorSpec ts = DCTensorSpec());

        // --- Accessors -------------------------------------------------------

        /**
         * @brief Get the coarse mesh (const reference).
         */
        const open3d::geometry::TriangleMesh & getCoarseMesh() const
        {
            return *coarse_mesh_;
        }

        /**
         * @brief Get the fine mesh (const reference).
         */
        const open3d::geometry::TriangleMesh & getFineMesh() const
        {
            return *fine_mesh_;
        }

        /**
         * @brief Get the list of convex collision pieces.
         */
        const std::vector<std::shared_ptr<const coal::CollisionGeometry>> & getConvexPieces() const
        {
            return convex_pieces_;
        }

        /**
         * @brief Get the bounding spheres tensor (shape: (N,4): x,y,z,r).
         *
         * The tensor follows the DCTensorSpec provided at construction (device / dtype).
         */
        const torch::Tensor & getBoundingSpheres() const
        {
            return bounding_spheres_;
        }

        /**
         * @brief Number of convex pieces.
         */
        int getNCvx() const
        {
            return n_cvx_;
        }

    private:
        /// @brief The coarse, watertight mesh used for visualization and sampling (no internal
        /// faces)
        std::shared_ptr<open3d::geometry::TriangleMesh> coarse_mesh_;

        /// @brief The fine-grained trimesh object, merged from convex pieces (has internal faces).
        std::shared_ptr<open3d::geometry::TriangleMesh> fine_mesh_;

        /// @brief List of convex collision geometries (from 'Coal' library), used for the
        /// narrow-phase (GJK/EPA) detection.
        std::vector<std::shared_ptr<const coal::CollisionGeometry>> convex_pieces_;

        /// @brief Bounding spheres (center_xyz, radius) for each convex piece, used for broad-phase
        /// culling.
        torch::Tensor bounding_spheres_;

        /// @brief Number of convex pieces in the mesh. Automatically computed during
        /// initialization.
        int n_cvx_ = 0;

        // --- Internal helpers -----------------------------------------------

        /**
         * @brief Load a convex mesh from file and convert to DCMesh internals.
         *
         * Internal helper for fromFile when loading convex data.
         */
        static DCMesh
        loadConvexFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts);

        /**
         * @brief Load a (processed) concave mesh directory and assemble convex parts.
         *
         * Internal helper for fromFile when loading concave MeshProcess outputs.
         */
        static DCMesh
        loadConcaveFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts);

        /**
         * @brief Compute the minimum bounding sphere for a given mesh.
         *
         * @param mesh The input TriangleMesh.
         * @return Eigen::Vector4d The bounding sphere (center_x, center_y, center_z, radius).
         */
        static Eigen::Vector4d computeMinimumSphere(const open3d::geometry::TriangleMesh & mesh);
    };
} // namespace diffcoal

#include "diffcoal/utils/mesh_io.hxx"

#endif // ifndef __diffcoal_utils_mesh_io_hpp__