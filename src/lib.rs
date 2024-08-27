pub mod linear;
pub mod sinc;

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
    C: NextSample,
{
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.cvtr.next_sample(&mut || self.iter.next())
    }
}

trait NextSample {
    fn next_sample<F>(&mut self, f: &mut F) -> Option<f64>
    where
        F: FnMut() -> Option<f64>;
}

pub trait Convert {
    fn proc_iter<'a, I>(&'a mut self, iter: I) -> ConvertIter<'a, I, Self>
    where
        I: Iterator<Item = f64>,
        Self: Sized;

    fn proc_slice(&mut self, input: &[f64]) -> Vec<f64>;
}

impl<T: NextSample> Convert for T {
    #[inline]
    fn proc_iter<'a, I>(&'a mut self, iter: I) -> ConvertIter<'a, I, Self> {
        ConvertIter::new(iter, self)
    }

    #[inline]
    fn proc_slice(&mut self, input: &[f64]) -> Vec<f64> {
        let mut v = Vec::new();
        let mut iter = input.iter().map(|&x| x);
        let mut f = || iter.next();
        while let Some(s) = self.next_sample(&mut f) {
            v.push(s);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "display only"]
    fn test1() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let manager = linear::Manager::new(2.0);
        let mut cvtr = manager.converter();
        for s in cvtr.proc_slice(&samples) {
            println!("sample = {s}");
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test2() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let manager = sinc::Manager::with_raw(2.0, 16, 4, 5.0, 1.0);
        for s in manager
            .converter()
            .proc_iter(samples.into_iter())
            .skip(manager.latency())
        {
            println!("sample = {s}");
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test3() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let manager = sinc::Manager::with_order(2.0, 30.0, 16, 4);
        for s in manager
            .converter()
            .proc_iter(samples.into_iter())
            .skip(manager.latency())
        {
            println!("sample = {s}");
        }
    }
}
