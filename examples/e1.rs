use simple_src::{linear, Convert};

fn main() {
    let samples1 = [1.0, 2.0, 3.0, 4.0];
    let samples2 = [5.0, 6.0, 7.0, 8.0];
    let manager = linear::Manager::new(2.0);
    let mut cvtr = manager.converter();
    for s in cvtr.proc_slice(&samples1) {
        println!("{s}");
    }
    for s in cvtr.proc_slice(&samples2) {
        println!("{s}");
    }
}
