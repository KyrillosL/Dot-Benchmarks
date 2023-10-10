//Note that fortran should be installed, but it's a part of gcc, and openblas
//brew install openblas :)

extern crate blas_src;
extern crate packed_simd;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng; // 0.6.5
use packed_simd::f64x8;

type Precision = f64;
const VEC_SIZE: u64 = 1_000_000;
#[allow(non_upper_case_globals)]
const RANGE_f64: f64 = 1.0;
//const N_ITERATIONS: usize = 10;

fn generate_native_vec() -> Vec<Precision> {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0.0, RANGE_f64);
    let data: Vec<Precision> = (0..VEC_SIZE).map(|_| rng.sample(&range)).collect();
    return data;
}

fn naive_native_rust(vec1: &Vec<Precision>, vec2: &Vec<Precision>) -> Precision {
    let mut output = 0.0;
    for i in 0..VEC_SIZE {
        output += vec1[i as usize] * vec2[i as usize];
    }
    return output;
}

pub fn native_rust(vec1: &Vec<Precision>, vec2: &Vec<Precision>) -> Precision {
    //https://stackoverflow.com/questions/65984936/efficient-simd-dot-product-in-rust
    //https://rust.godbolt.org/z/xEY3v1
    return vec1
        .iter()
        .zip(vec2)
        .map(|(&vec1, &vec2)| vec1 * vec2)
        .map(f64::from)
        .sum();
}

pub fn native_rust2(vec1: &Vec<Precision>, vec2: &Vec<Precision>) -> Precision {
    return vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
}

pub fn simd_f64x8(vec1: &Vec<Precision>, vec2: &Vec<Precision>) -> Precision {
    return vec1
        .chunks_exact(8)
        .map(f64x8::from_slice_unaligned)
        .zip(vec2.chunks_exact(8).map(f64x8::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f64x8>()
        .sum();
}

//TODO
/*
pub fn par_iter(vec1 : &Vec<f64>, vec2 : &Vec<f64>) -> f64{
    // what is par_iter doing? splitting for multiple threads
    // https://github.com/rayon-rs/rayon/blob/master/src/iter/plumbing/README.md
    return vec1
        .chunks_exact(8)
        .map(f64x8::from_slice_unaligned)
        .zip(vec2.chunks_exact(8).map(f64x8::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f64x8>()
        .sum();
}
*/

//TODO -> PAR ITER !
pub fn rust_ndarray_blas(vec1: &Array1<Precision>, vec2: &Array1<Precision>) -> Precision {
    //Using Blas as a backup
    //https://stackoverflow.com/questions/54028589/what-is-the-fastest-way-to-calculate-the-dot-product-of-two-f64-vectors-in-rust
    let result = vec1.dot(vec2);
    return result;
}

fn dot_product_benchmarks(c: &mut Criterion) {
    println!("Vector size {}", VEC_SIZE);
    let vec1 = generate_native_vec();
    let vec2 = generate_native_vec();

    //Using ndarrays for so2
    //We could have used for the generation
    //let x = Array1::random(VEC_SIZE, Uniform::<f64>::new(0., RANGE_f64));
    //let y = Array1::random(VEC_SIZE, Uniform::<f64>::new(0., RANGE_f64));
    let x = Array1::from_vec(vec1.clone());
    let y = Array1::from_vec(vec2.clone());

    c.bench_function("Naive Native", |b| {
        b.iter(|| naive_native_rust(black_box(&vec1), black_box(&vec2)))
    });
    c.bench_function("Native 1", |b| {
        b.iter(|| native_rust(black_box(&vec1), black_box(&vec2)))
    });
    c.bench_function("Native 2", |b| {
        b.iter(|| native_rust2(black_box(&vec1), black_box(&vec2)))
    });
    c.bench_function("SIMD f64x8", |b| {
        b.iter(|| simd_f64x8(black_box(&vec1), black_box(&vec2)))
    });
    c.bench_function("Blas ndarray", |b| {
        b.iter(|| rust_ndarray_blas(black_box(&x), black_box(&y)))
    });

    //TODO :  par_iter parallelism + SIMD f64x8
    //blas > all only when VECTOR_SIZE is large.
}

criterion_group! {
  name = dot_products;
  config = Criterion::default().sample_size(10);
  targets = dot_product_benchmarks
}
criterion_main!(dot_products);
