use super::*;
use crate::{Adjustable, Async, FixedAsync, Resampler, WindowFunction};
use audioadapter_buffers::direct::SequentialSliceOfVecs;

const MODES: [SincInterpolationType; 4] = [
    SincInterpolationType::Nearest,
    SincInterpolationType::Linear,
    SincInterpolationType::Quadratic,
    SincInterpolationType::Cubic,
];
fn parameters(mode: SincInterpolationType) -> SincInterpolationParameters {
    SincInterpolationParameters::new(32, WindowFunction::BlackmanHarris2)
        .oversampling_factor(16)
        .interpolation(mode)
}
fn compare_async<T: Sample + AvxSample + SseSample + NeonSample>() {
    for mode in MODES {
        for channels in [1, 2, 4] {
            for ratio in [0.5, 1.0, 2.0] {
                for target in [ratio, ratio * 0.75, ratio * 1.25] {
                    let params = parameters(mode);
                    let mut regular =
                        Async::<T>::new_sinc(ratio, 2.0, &params, 512, channels, FixedAsync::Input)
                            .unwrap();
                    regular.set_resample_ratio(target, true).unwrap();
                    let frames_in = regular.input_frames_next();
                    let frames_out = regular.output_frames_next();
                    let input: Vec<Vec<T>> = (0..channels)
                        .map(|channel| {
                            (0..frames_in)
                                .map(|frame| {
                                    T::coerce(((frame * 47 + channel * 19) as f64 * 0.031).sin())
                                })
                                .collect()
                        })
                        .collect();
                    let mut expected = vec![vec![T::zero(); frames_out]; channels];
                    let result = regular
                        .process_into_buffer(
                            &SequentialSliceOfVecs::new(&input, channels, frames_in).unwrap(),
                            &mut SequentialSliceOfVecs::new_mut(
                                &mut expected,
                                channels,
                                frames_out,
                            )
                            .unwrap(),
                            None,
                        )
                        .unwrap();
                    assert_eq!(result, (frames_in, frames_out));
                    let mut explicit = SincPositions::<T>::new(ratio, channels, &params).unwrap();
                    let length = explicit.sinc_len();
                    let padded: Vec<Vec<T>> = input
                        .iter()
                        .map(|channel| {
                            let mut buf = vec![T::zero(); 2 * length];
                            buf.extend(channel);
                            buf
                        })
                        .collect();
                    // Same first-block coordinates as ordinary Async, before latency trimming.
                    let mut index = -((length - 1) as f64);
                    let mut step = 1.0 / ratio;
                    let increment = (1.0 / target - step) / frames_out as f64;
                    let positions: Vec<f64> = (0..frames_out)
                        .map(|_| {
                            step += increment;
                            index += step;
                            index
                        })
                        .collect();
                    let mut actual = vec![vec![T::zero(); frames_out]; channels];
                    explicit
                        .render(
                            &padded,
                            &positions,
                            &mut SequentialSliceOfVecs::new_mut(&mut actual, channels, frames_out)
                                .unwrap(),
                            0,
                        )
                        .unwrap();
                    assert_eq!(
                        actual, expected,
                        "{mode:?}, {channels} channels, {ratio} -> {target}"
                    );
                }
            }
        }
    }
}
#[test]
fn ordinary_async_ramps_match_exactly_f32() {
    compare_async::<f32>();
}
#[test]
fn ordinary_async_ramps_match_exactly_f64() {
    compare_async::<f64>();
}

#[test]
fn repeated_reverse_and_shuffled_positions_preserve_samples_and_output_offset() {
    for mode in MODES {
        for channels in [1, 2, 4] {
            let mut engine = SincPositions::<f64>::new(1.0, channels, &parameters(mode)).unwrap();
            let input: Vec<Vec<f64>> = (0..channels)
                .map(|ch| {
                    (0..512)
                        .map(|n| (n as f64 * 0.13 + ch as f64).sin())
                        .collect()
                })
                .collect();
            let positions: Vec<f64> = (0..64).map(|n| 128.0 + n as f64 * 0.5).collect();
            let mut forward = vec![vec![0.0; 64]; channels];
            engine
                .render(
                    &input,
                    &positions,
                    &mut SequentialSliceOfVecs::new_mut(&mut forward, channels, 64).unwrap(),
                    0,
                )
                .unwrap();
            let order: Vec<usize> = (0..64).rev().chain([0, 0, 63, 18, 18, 1, 0]).collect();
            let reordered: Vec<f64> = order.iter().map(|&n| positions[n]).collect();
            let frames = reordered.len() + 4;
            let mut actual = vec![vec![-42.0; frames]; channels];
            engine
                .render(
                    &input,
                    &reordered,
                    &mut SequentialSliceOfVecs::new_mut(&mut actual, channels, frames).unwrap(),
                    2,
                )
                .unwrap();
            for ch in 0..channels {
                assert_eq!(&actual[ch][..2], &[-42.0; 2]);
                assert_eq!(&actual[ch][frames - 2..], &[-42.0; 2]);
                for (n, &original) in order.iter().enumerate() {
                    assert_eq!(actual[ch][n + 2].to_bits(), forward[ch][original].to_bits());
                }
            }
        }
    }
}

fn refused(
    input: &[Vec<f32>],
    positions: &[f64],
    channels: usize,
    frames: usize,
    offset: usize,
    expected: SincPositionError,
) {
    let mut engine =
        SincPositions::<f32>::new(1.0, 2, &parameters(SincInterpolationType::Cubic)).unwrap();
    let mut output = vec![vec![42.0; frames]; channels];
    let before = output.clone();
    assert_eq!(
        engine.render(
            input,
            positions,
            &mut SequentialSliceOfVecs::new_mut(&mut output, channels, frames).unwrap(),
            offset
        ),
        Err(expected)
    );
    assert_eq!(output, before);
}
#[test]
fn full_preflight_refuses_invalid_final_positions_without_partial_output() {
    let input = vec![vec![0.5; 512]; 2];
    for bad in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -f64::MAX,
        -100.0,
        512.0,
        480.0,
    ] {
        refused(
            &input,
            &[128.25, bad],
            2,
            2,
            0,
            SincPositionError::Position(1),
        );
    }
    refused(
        &[vec![0.5; 512], vec![0.5; 160]],
        &[128.25, 129.0],
        2,
        2,
        0,
        SincPositionError::Position(0),
    );
    refused(
        &[vec![], vec![]],
        &[128.25],
        2,
        1,
        0,
        SincPositionError::Position(0),
    );
}
#[test]
fn channel_and_output_ranges_are_checked_before_writes() {
    let input = vec![vec![0.5; 512]; 2];
    refused(&input[..1], &[128.25], 2, 1, 0, SincPositionError::Channels);
    refused(&input, &[128.25], 1, 1, 0, SincPositionError::Channels);
    refused(&input, &[128.25], 2, 1, 1, SincPositionError::OutputRange);
    refused(
        &input,
        &[128.25],
        2,
        1,
        usize::MAX,
        SincPositionError::OutputRange,
    );
    refused(
        &input,
        &[],
        2,
        1,
        usize::MAX,
        SincPositionError::OutputRange,
    );
    let mut engine =
        SincPositions::<f32>::new(1.0, 2, &parameters(SincInterpolationType::Cubic)).unwrap();
    let mut output = vec![vec![]; 2];
    engine
        .render(
            &[vec![], vec![]],
            &[],
            &mut SequentialSliceOfVecs::new_mut(&mut output, 2, 0).unwrap(),
            0,
        )
        .unwrap();
}
#[test]
fn strict_kernel_footprint_and_low_oversampling_boundaries() {
    for mode in MODES {
        let factor = if matches!(
            mode,
            SincInterpolationType::Nearest | SincInterpolationType::Linear
        ) {
            1
        } else {
            2
        };
        let params = parameters(mode).oversampling_factor(factor);
        let mut engine = SincPositions::<f32>::new(1.0, 1, &params).unwrap();
        let mut output = vec![vec![42.0]];
        let input = vec![vec![0.5; 100]];
        let mut adapter = SequentialSliceOfVecs::new_mut(&mut output, 1, 1).unwrap();
        // Every selected kernel start must have strictly more than L samples left.
        assert!(engine.render(&input, &[1.0], &mut adapter, 0).is_ok());
        assert!(engine.render(&input, &[4.0], &mut adapter, 0).is_err());
        assert!(
            engine.render(&input, &[-64.0], &mut adapter, 0).is_ok()
                == (mode != SincInterpolationType::Cubic)
        );
    }
}
#[test]
fn invalid_configuration_is_refused_before_filter_allocation() {
    let base = parameters(SincInterpolationType::Cubic);
    for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        assert!(SincPositions::<f32>::new(ratio, 2, &base).is_err());
    }
    for channels in [0, usize::MAX] {
        assert!(SincPositions::<f64>::new(1.0, channels, &base).is_err());
    }
    for len in [0, usize::MAX, usize::MAX / 2] {
        assert!(SincPositions::<f64>::new(1.0, 2, &base.sinc_len(len)).is_err());
    }
    for factor in [0, 1, usize::MAX] {
        assert!(SincPositions::<f64>::new(1.0, 2, &base.oversampling_factor(factor)).is_err());
    }
    for cutoff in [f32::NAN, f32::INFINITY, 0.0, -1.0, 1.01] {
        assert!(SincPositions::<f32>::new(0.5, 2, &base.f_cutoff(cutoff)).is_err());
    }
    assert_eq!(
        SincPositions::<f32>::new(1.0, 2, &base.sinc_len(25))
            .unwrap()
            .sinc_len(),
        32
    );
}
