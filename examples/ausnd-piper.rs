/*!

Example to show how to use AuStreamParser.
It reads AU data from stdin, adds volume and noise effects and writes AU data to stdout.

Example run, volume level 0.8 and noise level 0.1:

    cargo run --example ausnd-piper -- -v0.8 -n0.1 < input.au > output.au

To use with ffmpeg:

    ffmpeg -i music.mp3 -f au - | cargo run --example ausnd-piper -- -v1.5 -n0.2 | ffmpeg -i - -y out.mp3

*/
use std::env;
use std::io::{stdin, stdout, Read, Write};
use ausnd::{AuEvent, AuWriteInfo, AuStreamParser};

fn main() {
    // read effect options from command line arguments
    let mut volume_level = 1.0;
    let mut noise_level = 0.0;
    for arg in env::args() {
        if arg.starts_with("-v") {
            volume_level = arg[2..].parse().expect("Invalid volume value, should be a float");
        } else if arg.starts_with("-n") {
            noise_level = arg[2..].parse().expect("Invalid noise value, should be a float");
        }
    }

    // create a writer place-holder, it is still None because we don't have a header.
    let mut writer: Option<ausnd::AuWriter<std::io::Stdout>> = None;

    // create a state for the pseudo-random number generator
    let mut rng: u32 = 1;

    // streamer calls the callback when it has parsed some audio data
    let mut streamer = AuStreamParser::new(|event| {
        match event {
            AuEvent::Header(h) => {
                // create a new AuWriter to write to stdout using the same sample rate and channels,
                // but change the sample format to f32
                let stdout = stdout();
                let winfo = AuWriteInfo {
                    channels: h.channels,
                    sample_rate: h.sample_rate,
                    sample_format: ausnd::SampleFormat::F32,
                };
                writer = Some(ausnd::AuWriter::new(stdout, &winfo)
                    .expect("Writer failed"));
            },
            AuEvent::DescData(_) => { /* ignored */ },
            AuEvent::SampleData(samples) => {
                // add effects and write to stdout
                if let Some(w) = &mut writer {
                    for s in samples {
                        let sf32 = effect(simple_sample_to_f32(&s),
                            volume_level, noise_level, &mut rng);
                        w.write_samples_f32(&[ sf32 ]).expect("stdout write failed");
                    }
                }
            },
            AuEvent::End => {
                // end of stream - no need to do anything
                // we could finalize the writer here, but stdout doesn't implement Seek,
                // so we can't.
            }
        }
    });

    let mut stdin = stdin();

    // reads data from stdin and writes it to streamer - it uses a buffer, read() and write(),
    // but it could also just use std::io::copy():
    //std::io::copy(&mut stdin, &mut streamer).expect("Copy failed");
    let mut buf = [0u8; 32];
    loop {
        match stdin.read(&mut buf) {
            Ok(size) => {
                if size == 0 {
                    // end-of-stream, so break out of the loop
                    break;
                }
                streamer.write(&buf[0..size]).expect("Write error");
            },
            Err(e) => {
                eprintln!("ERROR: {:?}", e);
            },
        }
    }

    // after end-of-stream, dispatch the End event - it isn't really used for anything,
    // but this is just to show how to do it..
    streamer.finalize();
}

/// Super-simple effect function to adjust volume and to add noise.
fn effect(value: f32, volume: f32, noise: f32, rng: &mut u32) -> f32 {
    value * volume + noise * pseudo_random(rng)
}

/// Pseudo-random number generator "Park-Miller RNG".. we should use some real rng here..
fn pseudo_random(rng: &mut u32) -> f32 {
    *rng = rng.overflowing_mul(48271).0 % 0x7fff_ffff;
    *rng as f32 / 0x8000_0000u32 as f32 * 2.0 - 1.0
}

/// Simple conversion algorithm to convert Sample to f32
/// (integers are converted to [-1, 0.9921875]).
fn simple_sample_to_f32(s: &ausnd::Sample) -> f32 {
    match s {
        ausnd::Sample::I8(s) => { *s as f32 / (1u32 << 7) as f32 },
        ausnd::Sample::I16(s) => { *s as f32 / (1u32 << 15) as f32 },
        ausnd::Sample::I24(s) => { *s as f32 / (1u32 << 23) as f32 },
        ausnd::Sample::I32(s) => { (*s as f64 / (1u32 << 31) as f64) as f32 },
        ausnd::Sample::F32(s) => { *s },
        ausnd::Sample::F64(s) => { *s as f32 },
    }
}
