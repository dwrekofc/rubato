//! The explicit-position hot path must not allocate or release heap storage.
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{SincInterpolationParameters, SincInterpolationType, SincPositions};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counted;
static WATCH: AtomicBool = AtomicBool::new(false);
static CHANGES: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static ALLOCATOR: Counted = Counted;
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCH.load(Ordering::Relaxed) {
            CHANGES.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if WATCH.load(Ordering::Relaxed) {
            CHANGES.fetch_add(1, Ordering::Relaxed);
        }
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if WATCH.load(Ordering::Relaxed) {
            CHANGES.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, size)
    }
}
#[test]
fn render_and_validation_use_only_prepared_storage() {
    // Single test in this binary: the global counter observes only this work.
    for mode in [
        SincInterpolationType::Nearest,
        SincInterpolationType::Linear,
        SincInterpolationType::Quadratic,
        SincInterpolationType::Cubic,
    ] {
        for channels in [1, 2, 4] {
            let params = SincInterpolationParameters::default()
                .sinc_len(32)
                .oversampling_factor(16)
                .interpolation(mode);
            let mut engine = SincPositions::<f32>::new(1.0, channels, &params).unwrap();
            let input = vec![vec![0.25; 1024]; channels];
            let mut data = vec![vec![0.0; 256]; channels];
            let mut output = SequentialSliceOfVecs::new_mut(&mut data, channels, 256).unwrap();
            let positions: Vec<f64> = (0..256).map(|i| 256.0 - i as f64 * 0.5).collect();
            CHANGES.store(0, Ordering::Relaxed);
            WATCH.store(true, Ordering::Relaxed);
            let valid = engine.render(&input, &positions, &mut output, 0);
            let refused = engine.render(&input, &[f64::NAN], &mut output, 0);
            WATCH.store(false, Ordering::Relaxed);
            assert!(valid.is_ok());
            assert!(refused.is_err());
            assert_eq!(
                CHANGES.load(Ordering::Relaxed),
                0,
                "{mode:?}, {channels} channels"
            );
        }
    }
}
