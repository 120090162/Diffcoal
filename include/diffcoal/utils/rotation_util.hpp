#ifndef __diffcoal_utils_rotation_util_hpp__
#define __diffcoal_utils_rotation_util_hpp__

#include <torch/torch.h>
#include <cstdlib>

namespace diffcoal
{
    void set_seed(int64_t seed)
    {
        torch::manual_seed(seed);
        if (torch::cuda::is_available())
        {
            torch::cuda::manual_seed_all(seed);
        }

        std::srand(static_cast<unsigned int>(seed));
    }
} // namespace diffcoal

#endif // __diffcoal_utils_rotation_util_hpp__