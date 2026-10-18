//! Checked explicit-position entry point over the existing sinc implementation.
use crate::asynchro::InnerResampler;
use crate::asynchro_sinc::{make_interpolator, resolve_cutoff, InnerSinc};
use crate::interpolation::{
    get_nearest_time, get_nearest_times_2, get_nearest_times_3, get_nearest_times_4,
};
use crate::sinc_interpolator::{AvxSample, NeonSample, SincInterpolator, SseSample};
use crate::{Sample, SincInterpolationParameters, SincInterpolationType};
use audioadapter::AdapterMut;
use std::{alloc::Layout, fmt};

/// Validation failure before any output sample is written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SincPositionError {
    /// An invalid constructor parameter.
    Configuration(&'static str),
    /// Input or output channel count differs from the constructor.
    Channels,
    /// The requested output range does not fit.
    OutputRange,
    /// The frame's position or complete sinc footprint is invalid.
    Position(usize),
}
impl fmt::Display for SincPositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sinc position validation failed: {self:?}")
    }
}
impl std::error::Error for SincPositionError {}

/// Interpolate at explicit sinc-window coordinates using the ordinary sinc DSP.
///
/// A position is the same internal interpolation index used by `Async`, not a
/// decoded-frame playhead or a latency-corrected sample centre. Each selected
/// kernel starts at its integer index plus twice the actual sinc length in the
/// supplied input buffer. Keeping this coordinate avoids changing rounding in
/// existing `Async` trajectories. The caller owns input padding, filter/cutoff
/// policy, source timing and zero-speed muting.
/// Positions may repeat or move backward. All positions and complete input
/// footprints are checked before the first output write. No allocation occurs
/// in `render`; constructor and caller-owned input/output storage allocate.
pub struct SincPositions<T>
where
    T: Sample + AvxSample + SseSample + NeonSample,
{
    inner: InnerSinc<T>,
    channels: usize,
    mask: Vec<bool>,
    padding: isize,
}
impl<T> SincPositions<T>
where
    T: Sample + AvxSample + SseSample + NeonSample,
{
    /// Construct the same filter/interpolation implementation used by `Async`.
    /// `resample_ratio` controls filter design only; rendering uses positions.
    /// Choose the cutoff for the fastest input step to avoid downsampling aliasing.
    /// This entry point neither changes cutoff during rendering nor mutes repeated
    /// positions. Allocation failure follows the ordinary constructor behavior.
    pub fn new(
        resample_ratio: f64,
        channels: usize,
        parameters: &SincInterpolationParameters,
    ) -> Result<Self, SincPositionError> {
        let invalid = SincPositionError::Configuration;
        if !resample_ratio.is_finite() || resample_ratio <= 0.0 {
            return Err(invalid("ratio must be finite and positive"));
        }
        if channels == 0 || Layout::array::<bool>(channels).is_err() {
            return Err(invalid("channel count must be positive"));
        }
        let length = parameters
            .sinc_len
            .checked_add(7)
            .map(|n| n / 8 * 8)
            .filter(|n| *n > 0)
            .ok_or(invalid("invalid sinc length"))?;
        let factor = parameters.oversampling_factor;
        let minimum = if matches!(
            parameters.interpolation,
            SincInterpolationType::Cubic | SincInterpolationType::Quadratic
        ) {
            2
        } else {
            1
        };
        if factor < minimum
            || length
                .checked_mul(factor)
                .is_none_or(|n| Layout::array::<T>(n).is_err())
            || Layout::array::<Vec<T>>(factor).is_err()
            || length
                .checked_add(1)
                .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
                .is_none_or(|n| Layout::from_size_align(n, 32).is_err())
        {
            return Err(invalid("invalid sinc table dimensions"));
        }
        let padding = isize::try_from(
            length
                .checked_mul(2)
                .ok_or(invalid("invalid sinc length"))?,
        )
        .map_err(|_| invalid("invalid sinc length"))?;
        if parameters
            .f_cutoff
            .is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 1.0)
        {
            return Err(invalid("invalid cutoff"));
        }
        let cutoff = resolve_cutoff(
            length,
            resample_ratio,
            parameters.f_cutoff,
            parameters.window,
        );
        if !cutoff.is_finite() || cutoff <= 0.0 || cutoff > 1.0 {
            return Err(invalid("invalid cutoff"));
        }
        let interpolator = make_interpolator::<T>(
            length,
            resample_ratio,
            parameters.f_cutoff,
            factor,
            parameters.window,
        );
        Ok(Self {
            inner: InnerSinc::new(interpolator, parameters.interpolation),
            channels,
            mask: vec![true; channels],
            padding,
        })
    }
    /// Actual rounded filter length, useful for caller-owned padding.
    pub fn sinc_len(&self) -> usize {
        self.inner.nbr_points()
    }
    fn index(
        &self,
        position: f64,
        available: usize,
        frame: usize,
    ) -> Result<f64, SincPositionError> {
        let invalid = SincPositionError::Position(frame);
        let index = position;
        let factor = self.inner.interpolator.nbr_sincs() as isize;
        // Keep conversions and neighbour additions well inside signed-index range.
        let subposition = index * factor as f64;
        if !position.is_finite()
            || !subposition.is_finite()
            || subposition.abs() > (isize::MAX / 4) as f64
        {
            return Err(invalid);
        }
        let mut points = [(0isize, 0isize); 4];
        let count = match self.inner.interpolation {
            SincInterpolationType::Cubic => {
                get_nearest_times_4(index, factor, &mut points);
                4
            }
            SincInterpolationType::Quadratic => {
                let mut nearest = [(0, 0); 3];
                get_nearest_times_3(index, factor, &mut nearest);
                points[..3].copy_from_slice(&nearest);
                3
            }
            SincInterpolationType::Linear => {
                let mut nearest = [(0, 0); 2];
                get_nearest_times_2(index, factor, &mut nearest);
                points[..2].copy_from_slice(&nearest);
                2
            }
            SincInterpolationType::Nearest => {
                points[0] = get_nearest_time(index, factor);
                1
            }
        };
        for &(start, subindex) in &points[..count] {
            let start = start
                .checked_add(self.padding)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid.clone())?;
            // This is the strict bound used by the imported sinc interpolator.
            if subindex < 0
                || subindex >= factor
                || start
                    .checked_add(self.sinc_len())
                    .is_none_or(|end| end >= available)
            {
                return Err(invalid);
            }
        }
        Ok(index)
    }
    /// Render complete checked positions. Failed validation leaves output intact.
    pub fn render(
        &mut self,
        input: &[Vec<T>],
        positions: &[f64],
        output: &mut dyn AdapterMut<T>,
        output_offset: usize,
    ) -> Result<(), SincPositionError> {
        if input.len() != self.channels || output.channels() != self.channels {
            return Err(SincPositionError::Channels);
        }
        if output_offset
            .checked_add(positions.len())
            .is_none_or(|end| end > output.frames())
        {
            return Err(SincPositionError::OutputRange);
        }
        let available = input.iter().map(Vec::len).min().unwrap();
        for (frame, &position) in positions.iter().enumerate() {
            self.index(position, available, frame)?;
        }
        for (frame, &position) in positions.iter().enumerate() {
            // The existing processor advances index by zero, then performs its
            // unchanged SIMD/scalar sinc/interpolation and channel dispatch.
            self.inner.process(
                position,
                1,
                &self.mask,
                0.0,
                0.0,
                input,
                output,
                output_offset + frame,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
