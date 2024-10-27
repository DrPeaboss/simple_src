//! Simple sample rate conversion lib.
//!
//! ## Usage
//!
//! See [sinc] or [linear]

pub mod linear;
pub mod sinc;

mod ratio;
use ratio::{Ratio, Rational};

pub struct ConvertIter<'a, I, C> {
    iter: I,
    cvtr: &'a mut C,
}

impl<'a, I, C> ConvertIter<'a, I, C> {
    #[inline]
    pub fn new(iter: I, cvtr: &'a mut C) -> Self {
        Self { iter, cvtr }
    }
}

impl<I, C> Iterator for ConvertIter<'_, I, C>
where
    I: Iterator<Item = f64>,
    C: Convert,
{
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.cvtr.next_sample(&mut self.iter)
    }
}

pub trait Convert {
    /// Get the next sample converted, return `None` until the input samples is
    /// not enough.
    ///
    /// Note that the output can be continued after `None` returned.
    fn next_sample<I>(&mut self, iter: &mut I) -> Option<f64>
    where
        I: Iterator<Item = f64>,
        Self: Sized;

    /// Process samples and return an iterator, can be called multiple times.
    fn process<I>(&mut self, iter: I) -> ConvertIter<'_, I, Self>
    where
        I: Iterator<Item = f64>,
        Self: Sized,
    {
        ConvertIter::new(iter, self)
    }
}

#[derive(Debug)]
pub enum Error {
    UnsupportedRatio,
    InvalidParam,
    NotEnoughParam,
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct DynTest;

    impl DynTest {
        #[allow(dead_code)]
        pub fn new(a: i32) -> Box<dyn Convert> {
            if a == 0 {
                let manager = linear::Manager::new(2.0).unwrap();
                Box::new(manager.converter())
            } else {
                let manager = sinc::Manager::new(2.0, 48.0, 8, 0.2).unwrap();
                Box::new(manager.converter())
            }
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test1() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let manager = linear::Manager::new(2.0).unwrap();
        let mut cvtr = manager.converter();
        for s in cvtr.process(samples.into_iter()) {
            println!("sample = {s}");
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let manager = sinc::Manager::with_raw(2.0, 16, 4, 5.0, 1.0).unwrap();
        for s in manager
            .converter()
            .process(samples.into_iter())
            .skip(manager.latency())
        {
            println!("sample = {s}");
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test3() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let manager = sinc::Manager::with_order(2.0, 30.0, 16, 4).unwrap();
        for s in manager
            .converter()
            .process(samples.into_iter())
            .skip(manager.latency())
        {
            println!("sample = {s}");
        }
    }
}
