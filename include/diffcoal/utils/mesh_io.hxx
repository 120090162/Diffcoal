#ifndef __diffcoal_utils_mesh_io_hxx__
#define __diffcoal_utils_mesh_io_hxx__

#include <filesystem>

#include "diffcoal/collision/collision.hpp"
#include "diffcoal/utils/Miniball.hpp"

namespace diffcoal
{
    namespace fs = std::filesystem;
    inline DCMesh::DCMesh(
        std::shared_ptr<open3d::geometry::TriangleMesh> coarse,
        std::shared_ptr<open3d::geometry::TriangleMesh> fine,
        std::vector<std::shared_ptr<const coal::CollisionGeometry>> convex_parts,
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

        typedef std::vector<Eigen::Vector3d>::const_iterator PointIterator;
        typedef const double * CoordIterator;

        struct EigenAccessor
        {
            typedef PointIterator it;
            typedef CoordIterator const_ptr;
            const_ptr operator()(it i) const
            {
                return i->data();
            }
        };

        typedef Miniball::Miniball<EigenAccessor> MB;
        MB mb(3, hull_points.begin(), hull_points.end());

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

        std::vector<std::shared_ptr<const coal::CollisionGeometry>> cvx_lst = {
            getConvexFromData<diffcoal::context::Scalar, diffcoal::context::Options>(vertices)};
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
                fs::exists(simplified_path), "currently only support meshes processed by "
                                             "https://github.com/JYChen18/MeshProcess");
            return loadConcaveFromFileInternal(obj_path, scale, ts);
        }
    }

    inline DCMesh
    DCMesh::loadConvexFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts)
    {
        std::string final_path = fs::is_regular_file(path)
                                     ? path
                                     : (fs::path(path) / "mesh" / "simplified.obj").string();

        auto tm = open3d::io::CreateMeshFromFile(final_path);
        tm->Scale(scale, Eigen::Vector3d::Zero());

        Eigen::Vector4d sphere = computeMinimumSphere(*tm);
        torch::Tensor bs_raw = torch::from_blob(sphere.data(), {1, 4}, torch::kFloat64).clone();

        std::vector<std::shared_ptr<coal::CollisionGeometry>> cvx_lst = {
            getConvexFromFile(final_path, {scale, scale, scale})};

        return DCMesh(tm, tm, cvx_lst, ts.to(bs_raw));
    }

    inline DCMesh
    DCMesh::loadConcaveFromFileInternal(const std::string & path, double scale, DCTensorSpec & ts)
    {
        std::string tm_path = (fs::path(path) / "mesh" / "simplified.obj").string();
        std::string convex_dir = (fs::path(path) / "urdf" / "meshes").string();

        std::vector<std::string> files;
        for (const auto & entry : fs::directory_iterator(convex_dir))
        {
            if (entry.is_regular_file())
                files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());

        auto cm = open3d::io::CreateMeshFromFile(tm_path);
        cm->Scale(scale, Eigen::Vector3d::Zero());

        std::vector<Eigen::Vector4d> spheres;
        std::vector<std::shared_ptr<coal::CollisionGeometry>> cvx_lst;
        auto fm = std::make_shared<open3d::geometry::TriangleMesh>();
        std::vector<double> scale_vec = {scale, scale, scale};
        Eigen::Vector3d zero_vec = Eigen::Vector3d::Zero();
        for (const auto & f : files)
        {
            auto m = open3d::io::CreateMeshFromFile(f);
            m->Scale(scale, zero_vec);
            spheres.push_back(computeMinimumSphere(*m));
            cvx_lst.push_back(getConvexFromFile(f, scale_vec));
            *fm += *m;
        }

        torch::Tensor bs_tensor = torch::zeros({(long)spheres.size(), 4}, torch::kFloat64);
        for (size_t i = 0; i < spheres.size(); ++i)
        {
            std::memcpy(bs_tensor[i].data_ptr<double>(), spheres[i].data(), 4 * sizeof(double));
        }

        return DCMesh(cm, fm, cvx_lst, ts.to(bs_tensor));
    }

    inline Eigen::Vector4d DCMesh::computeMinimumSphere(const open3d::geometry::TriangleMesh & mesh)
    {
        if (mesh.vertices_.empty())
            return Eigen::Vector4d::Zero();

        // 1. 计算凸包 (对应 trimesh 源码中的 points = convex.hull_points(obj))
        // 注意：ComputeConvexHull 返回 std::tuple<mesh, indices>
        auto [hull_mesh, _] = mesh.ComputeConvexHull();
        const auto & hull_points = hull_mesh->vertices_;

        // 2. 配置 Miniball 访问器
        struct EigenAccessor
        {
            typedef std::vector<Eigen::Vector3d>::const_iterator it;
            typedef const double * const_ptr;
            const_ptr operator()(it i) const
            {
                return i->data();
            }
        };

        // 3. 计算最小包围球
        typedef Miniball::Miniball<EigenAccessor> MB;
        MB mb(3, hull_points.begin(), hull_points.end());

        // 4. 提取中心和半径
        Eigen::Vector3d center;
        const double * mb_center = mb.center();
        for (int i = 0; i < 3; ++i)
            center[i] = mb_center[i];
        double radius = std::sqrt(mb.squared_radius());

        Eigen::Vector4d res;
        res << center, radius;
        return res;
    }
} // namespace diffcoal

#endif // ifndef __diffcoal_utils_mesh_io_hxx__