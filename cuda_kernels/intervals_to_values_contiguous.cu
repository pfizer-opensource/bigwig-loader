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
        T* out
) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    // Same thread indexing as your original kernel
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
            T typed_value = cast_value<T>(value);

            // Output layout is now: batch_size x sequence_length x n_tracks
            // Position in flattened array: batch_index * (sequence_length * n_tracks) + pos * n_tracks + track_index
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
                T out_value;
                if (default_value_isnan) {
                    out_value = cast_value<T>(valid_count > 0 ? summation / valid_count : CUDART_NAN_F);
                } else {
                    summation = summation + (window_size - valid_count) * default_value;
                    out_value = cast_value<T>(summation / window_size);
                }

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
                T out_value;
                if (default_value_isnan) {
                    out_value = cast_value<T>(valid_count > 0 ? summation / valid_count : CUDART_NAN_F);
                } else {
                    summation = summation + (window_size - valid_count) * default_value;
                    out_value = cast_value<T>(summation / window_size);
                }

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

// Explicit instantiations with extern "C" wrapper
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
        float* out
) {
    intervals_to_values_kernel<float>(
        query_starts, query_ends, found_starts, found_ends,
        track_starts, track_ends, track_values,
        n_tracks, batch_size, sequence_length, max_number_intervals,
        window_size, default_value, default_value_isnan, out
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
        __nv_bfloat16* out
) {
    intervals_to_values_kernel<__nv_bfloat16>(
        query_starts, query_ends, found_starts, found_ends,
        track_starts, track_ends, track_values,
        n_tracks, batch_size, sequence_length, max_number_intervals,
        window_size, default_value, default_value_isnan, out
    );
}
