use crate::Convert;
use crate::kernel::{KernelBackend, linear, sinc};

pub(crate) enum ConverterInner {
    Linear(linear::Converter),
    Sinc(sinc::Converter),
}

/// Runtime sample-rate converter.
pub struct Converter {
    inner: ConverterInner,
}

impl Converter {
    pub(crate) fn from_backend(backend: &KernelBackend) -> Self {
        let inner = match backend {
            KernelBackend::Linear(b) => ConverterInner::Linear(b.converter()),
            KernelBackend::Sinc(b) => ConverterInner::Sinc(b.converter()),
        };
        Self { inner }
    }
}

impl Convert for Converter {
    #[inline]
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        match &mut self.inner {
            ConverterInner::Linear(c) => c.next_sample(iter),
            ConverterInner::Sinc(c) => c.next_sample(iter),
        }
    }

    fn flush(&mut self, output: &mut [f64]) -> usize
    where
        Self: Sized,
    {
        match &mut self.inner {
            ConverterInner::Linear(c) => c.flush(output),
            ConverterInner::Sinc(c) => c.flush(output),
        }
    }
}
