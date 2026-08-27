use crate::Convert;
use crate::kernel::KernelBackend;
use crate::kernel::spec::{KernelConverter, KernelSpec};

/// Runtime sample-rate converter.
pub struct Converter {
    inner: KernelConverter,
}

impl Converter {
    pub(crate) fn from_backend(backend: &KernelBackend) -> Self {
        Self {
            inner: backend.converter(),
        }
    }
}

impl Convert for Converter {
    /// Batch override: delegate straight to the kernel's batch path so the
    /// sample loop runs below the kernel dispatch.
    fn process_block(&mut self, input: &[f64], output: &mut [f64]) -> (usize, usize)
    where
        Self: Sized,
    {
        self.inner.process_block(input, output)
    }

    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        self.inner.next_sample(iter)
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        self.inner.flush(output)
    }
}
