use simple_src::{linear, Convert};

fn main() {
    let samples1 = [1.0, 2.0, 3.0, 4.0];
    let samples2 = [5.0, 6.0, 7.0, 8.0];
    let manager = linear::Manager::new(2.0).unwrap();
    let mut cvtr = manager.converter();
    for s in cvtr.process(samples1.into_iter()) {
        println!("{s}");
    }
    for s in cvtr.process(samples2.into_iter()) {
        println!("{s}");
    }
}
