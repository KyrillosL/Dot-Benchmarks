use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Fibo 5", |b| b.iter(|| fibonacci(black_box(5))));
    //If we would not use criterion_group!
    /*
    let mut group = c.benchmark_group("sample-size-example");
    group.sample_size(10);
    group.bench_function("Fibo 5", |b| b.iter(|| fibonacci(black_box(5))));
    group.finish();
    */
}



//criterion_group!(benches, criterion_benchmark); -> if we would use groups
criterion_group!{
  name = benches;
  config = Criterion::default().sample_size(10);
  targets = criterion_benchmark
}
criterion_main!(benches);
