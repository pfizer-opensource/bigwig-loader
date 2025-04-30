from bigwig_loader.sampler.position_sampler import RandomPositionSampler


def test_repeat_same_positions(merged_intervals):
    sampler = RandomPositionSampler(
        regions_of_interest=merged_intervals, repeat_same=True
    )

    first_samples = []
    for i, sample in enumerate(sampler):
        first_samples.append(sample)
        if i == 5:
            break
    second_samples = []
    for i, sample in enumerate(sampler):
        second_samples.append(sample)
        if i == 5:
            break

    assert first_samples == second_samples


def test_not_repeat_same_positions(merged_intervals):
    sampler = RandomPositionSampler(
        regions_of_interest=merged_intervals, repeat_same=False
    )

    first_samples = []
    for i, sample in enumerate(sampler):
        first_samples.append(sample)
        if i == 5:
            break
    second_samples = []
    for i, sample in enumerate(sampler):
        second_samples.append(sample)
        if i == 5:
            break

    assert first_samples != second_samples
