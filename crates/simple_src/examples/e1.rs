use simple_src::{Convert, SrcManager};

fn main() {
    let samples1 = [1.0, 2.0, 3.0, 4.0];
    let samples2 = [5.0, 6.0, 7.0, 8.0];
    let manager = SrcManager::with_ratio(2.0).unwrap();
    let mut cvtr = manager.converter();
    for s in cvtr.process(samples1.into_iter()) {
        println!("{s}");
    }
    for s in cvtr.process(samples2.into_iter()) {
        println!("{s}");
    }
}
