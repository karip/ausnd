use criterion::{criterion_group, criterion_main, black_box, Criterion};
use std::io::Cursor;
use ausnd::AuReader;

const DATA_SIZE: u32 = 2048;

fn create_au(data_size: u32, use_known_size: bool) -> Vec<u8> {
    let header_size: u32 = 24;
    let format: u32 = 06;
    let rate: u32 = 44100;
    let channels: u32 = 1;
    let desc: &[u8] = &[ 0, 0, 0, 0 ];
    let mut data = vec![ b'.', b's', b'n', b'd' ];  // magic
    data.extend_from_slice(&header_size.to_be_bytes());     // offset
    if use_known_size {
        data.extend_from_slice(&(data_size*4).to_be_bytes());           // data size
    } else {
        data.extend_from_slice(&[ 0xff, 0xff, 0xff, 0xff ]);        // unknown data size
    }
    data.extend_from_slice(&format.to_be_bytes());          // encoding 06, float
    data.extend_from_slice(&rate.to_be_bytes());            // sample rate
    data.extend_from_slice(&channels.to_be_bytes());        // channels
    data.extend_from_slice(&desc);                          // description
    // samples
    for _ in 0..data_size {
        data.extend_from_slice(&[ 0, 0, 0, 0 ]);
    }
    data
}

fn criterion_benchmark(c: &mut Criterion) {
    let audata_known = create_au(DATA_SIZE, true);
    let audata_unknown = create_au(DATA_SIZE, true);

    c.bench_function("known_samples_iter", |b| b.iter(|| {
        let cursor = Cursor::new(&audata_known);
        let mut reader = AuReader::new(cursor).expect("reader failed");
        for s in reader.samples().expect("no iterator") {
            black_box(match s.expect("iter failed") {
                ausnd::Sample::I8(_) => {},
                ausnd::Sample::I16(_) => {},
                ausnd::Sample::I24(_) => {},
                ausnd::Sample::I32(_) => {},
                ausnd::Sample::F32(_) => {},
                ausnd::Sample::F64(_) => {},
            })
        }
    }));

    c.bench_function("unknown_samples_iter", |b| b.iter(|| {
        let cursor = Cursor::new(&audata_unknown);
        let mut reader = AuReader::new(cursor).expect("reader failed");
        for s in reader.samples().expect("no iterator") {
            black_box(match s.expect("iter failed") {
                ausnd::Sample::I8(_) => {},
                ausnd::Sample::I16(_) => {},
                ausnd::Sample::I24(_) => {},
                ausnd::Sample::I32(_) => {},
                ausnd::Sample::F32(_) => {},
                ausnd::Sample::F64(_) => {},
            })
        }
    }));

    c.bench_function("known_read_sample", |b| b.iter(|| {
        let cursor = Cursor::new(&audata_known);
        let mut reader = AuReader::new(cursor).expect("reader failed");
        while let Some(s) = reader.read_sample().expect("sample error") {
            black_box(match s {
                ausnd::Sample::I8(_) => {},
                ausnd::Sample::I16(_) => {},
                ausnd::Sample::I24(_) => {},
                ausnd::Sample::I32(_) => {},
                ausnd::Sample::F32(_) => {},
                ausnd::Sample::F64(_) => {},
            })
        }
    }));

    c.bench_function("unknown_read_sample", |b| b.iter(|| {
        let cursor = Cursor::new(&audata_unknown);
        let mut reader = AuReader::new(cursor).expect("reader failed");
        while let Some(s) = reader.read_sample().expect("sample error") {
            black_box(match s {
                ausnd::Sample::I8(_) => {},
                ausnd::Sample::I16(_) => {},
                ausnd::Sample::I24(_) => {},
                ausnd::Sample::I32(_) => {},
                ausnd::Sample::F32(_) => {},
                ausnd::Sample::F64(_) => {},
            })
        }
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
