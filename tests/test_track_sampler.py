from bigwig_loader.sampler.track_sampler import TrackSampler


def test_track_samler():
    sampler = TrackSampler(total_number_of_tracks=40, sample_size=10)
    samples = []
    for i, sample in enumerate(sampler):
        samples.append(sample)
        assert len(sample) == 10
        assert all([0 <= s < 40 for s in sample])
        if i == 100:
            break
    assert len(samples) == 101
