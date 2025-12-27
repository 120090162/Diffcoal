#ifndef __diffcoal_utils_mesh_io_hxx__
#define __diffcoal_utils_mesh_io_hxx__

#include "diffcoal/collision/collision.hpp"
#include "diffcoal/utils/Miniball.hpp"

namespace diffcoal
{
    inline DCMesh::DCMesh(
        std::shared_ptr<open3d::geometry::TriangleMesh> coarse,
        std::shared_ptr<open3d::geometry::TriangleMesh> fine,
        std::vector<std::shared_ptr<coal::CollisionGeometry>> convex_parts,
        torch::Tensor spheres)
    : coarse_mesh_(std::move(coarse))
    , fine_mesh_(std::move(fine))
    , convex_pieces_(std::move(convex_parts))
    , bounding_spheres_(std::move(spheres))
    {
        n_cvx_ = static_cast<int>(convex_pieces_.size());
    }

    inline DCMesh DCMesh::fromData(
        const std::vector<Eigen::Vector3d> & vertices,
        const std::vector<Eigen::Vector3i> & faces,
        DCTensorSpec ts)
    {
        auto tm = std::make_shared<open3d::geometry::TriangleMesh>(vertices, faces);

        auto [hull_mesh, _] = tm->ComputeConvexHull();
        const auto & hull_points = hull_mesh->vertices_;
        struct EigenPointAccessor
        {
            typedef std::vector<Eigen::Vector3d>::const_iterator It;
            typedef double const * CoordinateIt;
            CoordinateIt operator()(It it) const
            {
                return it->data();
            }
        };

        Miniball::Miniball<EigenPointAccessor> mb(3, hull_points.begin(), hull_points.end());

        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        const double * mb_center = mb.center();
        for (int i = 0; i < 3; ++i)
        {
            center[i] = mb_center[i];
        }
        double radius = std::sqrt(mb.squared_radius());

        Eigen::Vector4d bs_vec;
        bs_vec << center, radius;

        torch::Tensor bs_raw = torch::from_blob(bs_vec.data(), {1, 4}, torch::kFloat64).clone();
        auto v_opts = torch::TensorOptions().dtype(torch::kFloat64);
        torch::Tensor v_tensor =
            torch::from_blob(
                const_cast<Eigen::Vector3d *>(vertices.data()), {(long)vertices.size(), 3}, v_opts)
                .to(torch::kFloat32);

        std::vector<std::shared_ptr<coal::CollisionGeometry>> cvx_lst = {
            getConvexFromData(v_tensor)};
        return DCMesh(hull_mesh, hull_mesh, cvx_lst, ts.to(bs_raw));
    }

    inline DCMesh
    DCMesh::fromFile(const std::string & obj_path, double scale, bool convex_hull, DCTensorSpec ts)
    {
        if (convex_hull)
        {
            return loadConvexFromFileInternal(obj_path, scale, ts);
        }
        else
        {
            std::string simplified_path = (fs::path(obj_path) / "mesh" / "simplified.obj").string();
            TORCH_CHECK(
                fs::exists(simplified_path),
                "[DCMesh] Missing simplified.obj for concave mesh at: ", simplified_path);
            return loadConcaveFromFileInternal(obj_path, scale, ts);
        }
    }

    inline Eigen::Vector4d
    DCMesh::computeBoundingSphereInternal(const open3d::geometry::TriangleMesh & mesh)
    {
        if (mesh.vertices_.empty())
            return Eigen::Vector4d::Zero();
        auto aabb = mesh.GetAxisAlignedBoundingBox();
        Eigen::Vector3d center = aabb.GetCenter();
        double max_dist_sq = 0;
        for (const auto & v : mesh.vertices_)
        {
            max_dist_sq = std::max(max_dist_sq, (v - center).squared_norm());
        }
        Eigen::Vector4d sphere;
        sphere << center, std::sqrt(max_dist_sq);
        return sphere;
    }

    inline DCMesh
    DCMesh::loadConvexFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts)
    {
        std::string final_path = fs::is_regular_file(path)
                                     ? path
                                     : (fs::path(path) / "mesh" / "simplified.obj").string();

        auto tm = open3d::io::CreateMeshFromFile(final_path);
        tm->Scale(scale, tm->GetCenter());
        auto [hull_mesh, _] = tm->ComputeConvexHull();

        Eigen::Vector4d sphere = computeBoundingSphereInternal(*hull_mesh);
        torch::Tensor sphere_lst = torch::from_blob(sphere.data(), {1, 4}, torch::kFloat64).clone();

        std::vector<std::shared_ptr<coal::CollisionGeometry>> cvx_lst = {
            getConvexFromFile(final_path, Eigen::Vector3d::Constant(scale))};

        return DCMesh(hull_mesh, hull_mesh, cvx_lst, ts.to(sphere_lst));
    }

    inline DCMesh
    DCMesh::loadConcaveFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts)
    {
        std::string tm_path = (fs::path(path) / "mesh" / "simplified.obj").string();
        std::string convex_dir = (fs::path(path) / "urdf" / "meshes").string();

        auto cm = open3d::io::CreateMeshFromFile(tm_path);
        cm->Scale(scale, cm->GetCenter());

        std::vector<std::string> files;
        for (const auto & entry : fs::directory_iterator(convex_dir))
        {
            if (entry.is_regular_file())
                files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());

        std::vector<Eigen::Vector4d> spheres;
        std::vector<std::shared_ptr<coal::CollisionGeometry>> cvx_lst;
        auto fm = std::make_shared<open3d::geometry::TriangleMesh>();

        for (const auto & f : files)
        {
            auto m = open3d::io::CreateMeshFromFile(f);
            m->Scale(scale, m->GetCenter());
            spheres.push_back(computeBoundingSphereInternal(*m));
            cvx_lst.push_back(getConvexFromFile(f, Eigen::Vector3d::Constant(scale)));
            *fm += *m;
        }

        torch::Tensor bs_tensor = torch::zeros({(long)spheres.size(), 4}, torch::kFloat64);
        for (size_t i = 0; i < spheres.size(); ++i)
        {
            std::memcpy(bs_tensor[i].data_ptr<double>(), spheres[i].data(), 4 * sizeof(double));
        }

        return DCMesh(cm, fm, cvx_lst, ts.to(bs_tensor));
    }
} // namespace diffcoal

#endif // ifndef __diffcoal_utils_mesh_io_hxx__