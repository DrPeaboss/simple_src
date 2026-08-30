#[path = "common/mod.rs"]
mod common;
use common::*;
use simple_src::SrcManager;

fn main() {
    divan::main();
}

#[divan::bench(
    name="1. init a96",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a96(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 96.0, false)
}

#[divan::bench(
    name="2. init a120",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a120(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 120.0, false)
}

#[divan::bench(
    name="3. init a144",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a144(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 144.0, false)
}

#[divan::bench(
    name="1. init a96 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a96_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 96.0, true)
}

#[divan::bench(
    name="2. init a120 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a120_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 120.0, true)
}

#[divan::bench(
    name="3. init a144 fast",
    args=[Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k]
)]
fn init_a144_fast(conv: &Conv) -> SrcManager {
    sinc_manager(conv, 144.0, true)
}

#[divan::bench(
    name = "6. init bit16 quality generic",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
)]
fn init_quality_bit16_generic(conv: &Conv) -> SrcManager {
    quality_sinc_manager(conv, false)
}

#[divan::bench(
    name = "6. init bit16 quality fast",
    args = [Conv::C44k48k, Conv::C44k96k, Conv::C48k44k, Conv::C48k96k, Conv::C96k44k, Conv::C96k48k],
)]
fn init_quality_bit16_fast(conv: &Conv) -> SrcManager {
    quality_sinc_manager(conv, true)
}

#[divan::bench(
    name = "5. sinc generic shape init",
    args = [Shape::FloatPi, Shape::Generic20000Of19999, Shape::Up16, Shape::Down16],
)]
fn sinc_generic_shape_init(shape: &Shape) -> SrcManager {
    shape_sinc_manager(shape, false)
}

#[divan::bench(
    name = "5. sinc fast shape init",
    args = [Shape::Up16, Shape::Down16],
)]
fn sinc_fast_shape_init(shape: &Shape) -> SrcManager {
    shape_sinc_manager(shape, true)
}
