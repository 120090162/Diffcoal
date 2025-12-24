#ifndef __diffcoal_utils_openmp_hpp__
#define __diffcoal_utils_openmp_hpp__

#include <cstdlib>
#include <exception>
#include <mutex>
#include <omp.h>

namespace diffcoal
{

    /// \brief Returns the number of threads defined by the environment variable OMP_NUM_THREADS.
    ///
    /// This helper reads the `OMP_NUM_THREADS` environment variable and converts it to an
    /// integer. If the variable is not present or cannot be parsed, the function returns
    /// a sensible default of 1. This is intended for simple configuration checks and
    /// should not be used to set OpenMP behaviour directly (use `setDefaultOpenMPSettings`).
    ///
    /// \returns Number of threads requested via the environment, or 1 if unset.
    inline int getOpenMPNumThreadsEnv()
    {
        int num_threads = 1;

        if (const char * env_p = std::getenv("OMP_NUM_THREADS"))
            num_threads = atoi(env_p);

        return num_threads;
    }

    /// \brief Configure common OpenMP settings used by the library.
    ///
    /// This function sets the number of OpenMP threads to `num_threads` and disables
    /// dynamic adjustment of the number of threads (`omp_set_dynamic(0)`) so that the
    /// runtime will try to use the requested thread count. The default argument uses
    /// `omp_get_max_threads()` as a sensible default for the host machine.
    ///
    /// \param num_threads Desired number of OpenMP threads (default: omp_get_max_threads()).
    inline void setDefaultOpenMPSettings(const size_t num_threads = (size_t)omp_get_max_threads())
    {
        omp_set_num_threads((int)num_threads);
        omp_set_dynamic(0);
    }

    /// \brief Helper to capture and propagate exceptions thrown inside OpenMP regions.
    ///
    /// OpenMP often executes work in parallel threads where throwing exceptions across
    /// thread boundaries is not directly supported. `OpenMPException` provides a simple
    /// mechanism to capture the first exception that occurs in a worker thread and
    /// rethrow it later (for example, in the calling thread) once the parallel region
    /// has completed.
    ///
    /// Usage pattern:
    /// - Create an instance before entering a parallel region.
    /// - In each worker, call `run(...)` to execute a callable; if it throws,
    ///   the exception is captured.
    /// - After the parallel region, check `hasThrown()` or call `rethrowException()` to
    ///   propagate the captured exception into the current thread.
    ///
    /// Thread-safety: `captureException()` uses a mutex to serialize access to the
    /// stored `std::exception_ptr`. Only the first captured exception is stored.
    struct OpenMPException
    {
        /// \param throw_on_deletion If true, the destructor will rethrow the captured
        /// exception when the object is destroyed. Prefer explicit `rethrowException()`
        /// in code to make control flow clearer; this option can be useful for tests.
        explicit OpenMPException(const bool throw_on_deletion = false)
        : m_exception_ptr(nullptr)
        , m_throw_on_deletion(throw_on_deletion)
        {
        }

        /// \brief Rethrows the stored exception in the current thread, if any.
        ///
        /// Call this after a parallel region to propagate the first exception that
        /// occurred in a worker thread into the caller.
        void rethrowException() const
        {
            if (this->m_exception_ptr)
                std::rethrow_exception(this->m_exception_ptr);
        }

        /// \brief Returns the raw `std::exception_ptr` currently stored (may be null).
        std::exception_ptr getException() const
        {
            return m_exception_ptr;
        }

        /// \brief True if an exception has been captured.
        bool hasThrown() const
        {
            return this->m_exception_ptr != nullptr;
        }

        /// \brief Executes `f(params...)` and captures any exception thrown.
        ///
        /// This method is intended to be called from worker threads. It will swallow
        /// the exception in the worker and store it for later rethrowing.
        template<typename Function, typename... Parameters>
        void run(Function f, Parameters... params)
        {
            try
            {
                f(params...);
            }
            catch (...)
            {
                this->captureException();
            }
        }

        /// \brief Capture the current exception into `m_exception_ptr`.
        ///
        /// Uses a mutex (`m_lock`) to ensure only one thread stores the exception
        /// pointer reliably. Calling multiple times will overwrite the previous
        /// stored pointer; callers should treat only the first meaningful exception
        /// as the root cause.
        void captureException()
        {
            std::unique_lock<std::mutex> guard(this->m_lock);
            this->m_exception_ptr = std::current_exception();
        }

        /// \brief Reset the stored exception pointer to null.
        void reset()
        {
            this->m_exception_ptr = nullptr;
        }

        /// \brief Destructor optionally rethrows the stored exception.
        ///
        /// Rethrowing in a destructor is generally discouraged because it can
        /// terminate the program if another exception is active. This behaviour is
        /// therefore opt-in via the constructor parameter `throw_on_deletion`.
        ~OpenMPException()
        {
            if (m_throw_on_deletion)
                this->rethrowException();
        }

    protected:
        std::exception_ptr m_exception_ptr;
        std::mutex m_lock;
        const bool m_throw_on_deletion;
    }; // struct OpenMPException
} // namespace diffcoal

#endif // ifndef __diffcoal_utils_openmp_hpp__
