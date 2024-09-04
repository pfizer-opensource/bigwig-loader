extern "C" __global__
void intervals_to_values(
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
        float* out
) {

    int thread = blockIdx.x * blockDim.x + threadIdx.x;

//     # out is a 1D array of size batch_size x n_tracks x sequence_length
//
//     # n_tracks x n_batch

    int batch_index = thread % batch_size;
    int i = thread % (batch_size * n_tracks);

    if (window_size == 1){
		int j = (thread / (batch_size * n_tracks)) % max_number_intervals;

		int found_start_index = found_starts[i];
		int found_end_index = found_ends[i];
		int query_start = query_starts[batch_index];
		int query_end = query_ends[batch_index];

		int cursor = found_start_index + j;

		if (cursor < found_end_index){
			int interval_start = track_starts[cursor];
			int interval_end = track_ends[cursor];
			int start_index = max(interval_start - query_start, 0);
			int end_index = (i * sequence_length) + min(interval_end, query_end) - query_start;
			int start_position = (i * sequence_length) + start_index;

			float value = track_values[cursor];

			for (int position = start_position; position < end_index; position++){
				out[position] = value;
			}
		}
	} else {

		int track_index = i / batch_size;

		int found_start_index = found_starts[i];
		int found_end_index = found_ends[i];
		int query_start = query_starts[batch_index];
		int query_end = query_ends[batch_index];

		int cursor = found_start_index;
		int window_index = 0;
		float summation = 0.0f;

		int reduced_dim = sequence_length / window_size;

		while (cursor < found_end_index && window_index < reduced_dim) {
			int window_start = window_index * window_size;
			int window_end = window_start + window_size;

			int interval_start = track_starts[cursor];
			int interval_end = track_ends[cursor];

			int start_index = max(interval_start - query_start, 0);
			int end_index = min(interval_end, query_end) - query_start;

			if (start_index >= window_end) {
				out[i * reduced_dim + window_index] = summation / window_size;
				summation = 0.0f;
				window_index += 1;
				continue;
			}

			int number = min(window_end, end_index) - max(window_start, start_index);

			summation += number * track_values[cursor];

			if (end_index >= window_end || cursor + 1 >= found_end_index) {
				out[i * reduced_dim + window_index] = summation / window_size;
				summation = 0.0f;
				window_index += 1;
			}

			if (end_index < window_end) {
				cursor += 1;
			}
		}
	}
}
