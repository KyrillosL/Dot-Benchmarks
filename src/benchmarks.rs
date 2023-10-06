//Note that fortran should be installed, but it's a part of gcc, and openblas
//brew install openblas :)

extern crate blas_src;
//extern crate time;
use std::time::Instant;
use rand::{distributions::Uniform as uni_test, Rng}; // 0.6.5
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::time::Duration;

//#![feature(portable_simd)]
//use std::simd::f64x8;
extern crate packed_simd;
use packed_simd::{f64x8, FromCast};




type Precision = f64;
//const VEC_SIZE: usize = usize::MAX / 1000000000000;
const VEC_SIZE: u64 = 1_000_000;
const PROGRESS_BAR_RANGE : u64 = VEC_SIZE;
const N_ITERATIONS: usize = 10;
const RANGE_f64: f64 = 1.0;

fn generate_native_vec() -> Vec<f64>{
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0.0, RANGE_f64);
    let data: Vec<f64> = (0..VEC_SIZE).map(|_| rng.sample(&range)).collect();
    return data;
}

fn naive_native_rust(vec1 : &Vec<f64>, vec2 : &Vec<f64>) -> f64 {
    let mut output = 0.0;
    for i in 0..VEC_SIZE {
        output += vec1[i as usize] * vec2[i as usize];
    }
    return output;
}

pub fn native_rust(vec1 : &Vec<f64>, vec2 : &Vec<f64>) -> f64{
    //https://stackoverflow.com/questions/65984936/efficient-simd-dot-product-in-rust
    //https://rust.godbolt.org/z/xEY3v1
    return vec1.iter()
        .zip(vec2)
        .map(|(&vec1, &vec2)|vec1*vec2)
        .map(f64::from)
        .sum()
}

pub fn native_rust2(vec1 : &Vec<f64>, vec2 : &Vec<f64>) -> f64{
    return vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| a * b)
        .sum();
}

pub fn simd_f64x8(vec1 : &Vec<f64>, vec2 : &Vec<f64>) -> f64{
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
pub fn rust_ndarray_blas(vec1 : &Array1<f64>, vec2 : &Array1<f64>) -> f64{
    //Using Blas as a backup
    //https://stackoverflow.com/questions/54028589/what-is-the-fastest-way-to-calculate-the-dot-product-of-two-f64-vectors-in-rust
    let result = vec1.dot(vec2);
    return result;
}



pub(crate) fn benchmark_fn<T>(vec1 : &T, vec2 : &T, f: &dyn Fn(&T, &T) -> f64) -> Vec<Duration> {

    let mut results : Vec<Duration> = Vec::new();
    for _i in 0..N_ITERATIONS {
        //let start_time = PreciseTime::now();
        let start_time = Instant::now();
        let result : f64 = f(&vec1, &vec2);
        //f(&vec1, &vec2);
        let end_time = Instant::now();
        //let elapsed_time = start_time.to(end_time); //Precise
        let elapsed_time = end_time - start_time;
        results.push(elapsed_time);
        //print!("{}    ", result);
        //print!("test");

    }
    //println!("time: {:#?}, result : {:#?}", elapsed_time, result);
    println!("time: {:#?}", results);
    return results;
}


pub(crate) fn compute_benchmarks() {

    println!("Vector size {}", VEC_SIZE);
    let vec1 = generate_native_vec();
    let vec2 = generate_native_vec();

    //Using ndarrays for so2
    //We could have used for the generation
    //let x = Array1::random(VEC_SIZE, Uniform::<f64>::new(0., RANGE_f64));
    //let y = Array1::random(VEC_SIZE, Uniform::<f64>::new(0., RANGE_f64));
    let x = Array1::from_vec(vec1.clone());
    let y = Array1::from_vec(vec2.clone());

    //results - release : : ~7ms, release : ~26ms
    print!("naive native: ");
    let r = benchmark_fn(&vec1, &vec2, &naive_native_rust);

    //results - release : : ~4ms, release : ~24ms
    print!("native_rust: ");
    benchmark_fn(&vec1, &vec2, &native_rust);

    //results - release : : ~4ms, release : ~24ms
    print!("native_rust2: ");
    benchmark_fn(&vec1, &vec2, &native_rust2);

    //results - release : : ~4ms, release : ~24ms
    print!("SIMD f64x8: ");
    benchmark_fn(&vec1, &vec2, &simd_f64x8);


    //TODO
    /*
    //results - release : : ~4ms, release : ~24ms
    print!("par_iter parallelism + SIMD f64x8: ");
    benchmark_fn(&vec1, &vec2, &par_iter);
    */

    //results - release : ~18ms
    print!("open_blas - openblas: ");
    benchmark_fn(&x, &y, &rust_ndarray_blas);

    //blas > all only when VECTOR_SIZE is large.
}
