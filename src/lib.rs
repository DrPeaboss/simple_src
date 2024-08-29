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
    fn process<I>(&mut self, iter: I) -> ConvertIter<'_, I, Self>
    where
        I: Iterator<Item = f64>,
        Self: Sized;
}

impl<T: NextSample> Convert for T {
    #[inline]
    fn process<I>(&mut self, iter: I) -> ConvertIter<'_, I, Self> {
        ConvertIter::new(iter, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    struct DynTest;

    impl DynTest {
        #[allow(dead_code)]
        pub fn new(a: i32) -> Box<dyn Convert> {
            if a == 0 {
                Box::new(linear::Converter::new(0.5))
            } else {
                let filter = std::sync::Arc::new(vec![]);
                Box::new(sinc::Converter::new(0.5, 128, 128, filter))
            }
        }
    }

    #[test]
    #[ignore = "display only"]
    fn test1() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let manager = linear::Manager::new(2.0);
        let mut cvtr = manager.converter();
        for s in cvtr.process(samples.into_iter()) {
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
        let manager = sinc::Manager::with_order(2.0, 30.0, 16, 4);
        for s in manager
            .converter()
            .process(samples.into_iter())
            .skip(manager.latency())
        {
            println!("sample = {s}");
        }
    }
}
