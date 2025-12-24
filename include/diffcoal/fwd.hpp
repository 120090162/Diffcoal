#ifndef __diffcoal_fwd_hpp__
#define __diffcoal_fwd_hpp__

#ifdef _WIN32
    #include <windows.h>
    #undef far
    #undef near
#endif

#include <cassert>

#ifdef DIFFCOAL_EIGEN_CHECK_MALLOC
    #ifndef EIGEN_RUNTIME_NO_MALLOC
        #define EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
        #define EIGEN_RUNTIME_NO_MALLOC
    #endif
#endif

#include "diffcoal/deprecated.hpp"
#include "diffcoal/warning.hpp"
#include "diffcoal/config.hpp"

// Include Eigen components
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace diffcoal
{
    ///
    /// \brief Common traits structure to fully define base classes for CRTP.
    ///
    template<typename C>
    struct traits
    {
    };

    namespace context
    {

        using Scalar = double;
        enum
        {
            Options = 0
        };

        // Common eigen types
        using Vector2s = Eigen::Matrix<Scalar, 2, 1, Options>;
        using Vector3s = Eigen::Matrix<Scalar, 3, 1, Options>;
        using Vector6s = Eigen::Matrix<Scalar, 6, 1, Options>;
        using Matrix6s = Eigen::Matrix<Scalar, 6, 6, Options>;
        using Matrix63s = Eigen::Matrix<Scalar, 6, 3, Options>;
        using Matrix6Xs = Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options>;
        using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
        using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
        using RowMatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options | Eigen::RowMajor>;

    } // namespace context
} // namespace diffcoal

#endif // __diffcoal_fwd_hpp__
