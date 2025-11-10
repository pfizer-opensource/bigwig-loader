#include <math_constants.h>
#include <cuda_bf16.h>

template<typename T>
__device__ __forceinline__ T cast_value(float value);

template<>
__device__ __forceinline__ float cast_value<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ __nv_bfloat16 cast_value<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template<typename T>
__global__
void intervals_to_values_kernel(
        const unsigned int* query_starts,
        const unsigned int* query_ends,
        const unsigned int* found_starts,
        const unsigned int* found_ends,
        const unsigned int* track_starts,
        const unsigned int* track_ends,
        const float* track_values,
        const int n_tracks,
        const int batch_size,
        const int sequence_length,
        const int max_number_intervals,
        const int window_size,
        const float default_value,
        const bool default_value_isnan,
        const float* scaling_factors,  // NEW: scaling factors array of length n_tracks
        T* out
) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    // Same thread indexing as before
    int batch_index = thread % batch_size;
    int i = thread % (batch_size * n_tracks);
    int track_index = i / batch_size;

    if (window_size == 1) {
        int j = (thread / (batch_size * n_tracks)) % max_number_intervals;

        int found_start_index = found_starts[i];
        int found_end_index = found_ends[i];
        int query_start = query_starts[batch_index];
        int query_end = query_ends[batch_index];

        int cursor = found_start_index + j;

        if (cursor < found_end_index) {
            int interval_start = track_starts[cursor];
            int interval_end = track_ends[cursor];
            int start_index = max(interval_start - query_start, 0);
            int end_index = min(interval_end, query_end) - query_start;

            float value = track_values[cursor];

            // Apply scaling factor here
            if (scaling_factors != nullptr) {
                value *= scaling_factors[track_index];
            }

            T typed_value = cast_value<T>(value);

            int base_offset = batch_index * (sequence_length * n_tracks) + track_index;

            for (int pos = start_index; pos < end_index; pos++) {
                out[base_offset + pos * n_tracks] = typed_value;
            }
        }
    } else {
        int found_start_index = found_starts[i];
        int found_end_index = found_ends[i];
        int query_start = query_starts[batch_index];
        int query_end = query_ends[batch_index];

        int cursor = found_start_index;
        int window_index = 0;
        float summation = 0.0f;
        int valid_count = 0;

        int reduced_dim = sequence_length / window_size;

        while (cursor < found_end_index && window_index < reduced_dim) {
            int window_start = window_index * window_size;
            int window_end = window_start + window_size;

            int interval_start = track_starts[cursor];
            int interval_end = track_ends[cursor];

            int start_index = max(interval_start - query_start, 0);
            int end_index = min(interval_end, query_end) - query_start;

            if (start_index >= window_end) {
                float final_value = (valid_count > 0) ? (summation / valid_count) : 0.0f;

                // Apply scaling factor
                if (scaling_factors != nullptr) {
                    final_value *= scaling_factors[track_index];
                }

                if (default_value_isnan && valid_count == 0) {
                    final_value = CUDART_NAN_F;
                }

                T out_value = cast_value<T>(final_value);
                int out_idx = batch_index * (reduced_dim * n_tracks) + window_index * n_tracks + track_index;
                out[out_idx] = out_value;

                summation = 0.0f;
                valid_count = 0;
                window_index += 1;
                continue;
            }

            int number = min(window_end, end_index) - max(window_start, start_index);

            if (number > 0) {
                summation += number * track_values[cursor];
                valid_count += number;
            }

            if (end_index >= window_end || cursor + 1 >= found_end_index) {
                float final_value = (valid_count > 0) ? (summation / valid_count) : 0.0f;

                // Apply scaling factor
                if (scaling_factors != nullptr) {
                    final_value *= scaling_factors[track_index];
                }

                if (default_value_isnan && valid_count == 0) {
                    final_value = CUDART_NAN_F;
                }

                T out_value = cast_value<T>(final_value);
                int out_idx = batch_index * (reduced_dim * n_tracks) + window_index * n_tracks + track_index;
                out[out_idx] = out_value;

                summation = 0.0f;
                valid_count = 0;
                window_index += 1;
            }

            if (end_index < window_end) {
                cursor += 1;
            }
        }
    }
}

// Update wrapper functions to include scaling_factors

extern "C" __global__
void intervals_to_values_float32(
        const unsigned int* query_starts,
        const unsigned int* query_ends,
        const unsigned int* found_starts,
        const unsigned int* found_ends,
        const unsigned int* track_starts,
        const unsigned int* track_ends,
        const float* track_values,
        const int n_tracks,
        const int batch_size,
        const int sequence_length,
        const int max_number_intervals,
        const int window_size,
        const float default_value,
        const bool default_value_isnan,
        const float* scaling_factors,
        float* out
) {
    intervals_to_values_kernel<float>(
        query_starts, query_ends, found_starts, found_ends,
        track_starts, track_ends, track_values,
        n_tracks, batch_size, sequence_length, max_number_intervals,
        window_size, default_value, default_value_isnan,
        scaling_factors, out
    );
}

extern "C" __global__
void intervals_to_values_bfloat16(
        const unsigned int* query_starts,
        const unsigned int* query_ends,
        const unsigned int* found_starts,
        const unsigned int* found_ends,
        const unsigned int* track_starts,
        const unsigned int* track_ends,
        const float* track_values,
        const int n_tracks,
        const int batch_size,
        const int sequence_length,
        const int max_number_intervals,
        const int window_size,
        const float default_value,
        const bool default_value_isnan,
        const float* scaling_factors,
        __nv_bfloat16* out
) {
    intervals_to_values_kernel<__nv_bfloat16>(
        query_starts, query_ends, found_starts, found_ends,
        track_starts, track_ends, track_values,
        n_tracks, batch_size, sequence_length, max_number_intervals,
        window_size, default_value, default_value_isnan,
        scaling_factors, out
    );
}
