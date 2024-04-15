/*!

Example to play AU audio files using ausnd and tinyaudio.

Example run:

    cargo run --example ausnd-tinyaudio filename.au

*/
use std::env;
use std::fs::File;
use std::io::BufReader;
use tinyaudio::prelude::*;

fn main() {

    // open au file for reading
    let filename = env::args().nth(1)
        .expect("Filename missing");
    let bufreader = BufReader::new(File::open(filename)
        .expect("Can't open file"));
    let mut reader = ausnd::AuReader::new(bufreader)
        .expect("Can't read the AU file");
    let info = reader.read_info()
        .expect("Can't read audio info from the file");

    if info.sample_rate == 0 || info.channels == 0 {
        println!("Audio file: {} channels, sample rate {}, format {:?}. Can't play this file.",
            info.channels, info.sample_rate, info.sample_format);
        return;
    }

    let total_samples = reader.resolved_sample_len().expect("Can't get total sample count");
    let duration = total_samples as f64 / info.channels as f64 / info.sample_rate as f64;
    println!("Audio file: {} channels, sample rate {}, format {:?}, duration {:.3} secs.",
        info.channels, info.sample_rate, info.sample_format, duration);

    // audio buffer size = total sample len or max 2048 bytes
    let channel_sample_count = 2048.min(total_samples as usize);

    // play the samples using tinyaudio
    let _device = run_output_device(
        OutputDeviceParameters {
            channels_count: info.channels as usize,
            sample_rate: info.sample_rate as usize,
            channel_sample_count,
        },
        move |data| {
            // read samples from the au file to fill the audio buffer
            // ideally, the samples should be buffered to avoid glitches in audio
            let mut siter = reader.samples().expect("Can't iterate");
            for sample in data.iter_mut() {
                *sample = match siter.next() {
                    Some(s) => simple_sample_to_f32(&s.expect("sample error")),
                    None => 0.0
                };
            }
        },
    ).expect("Audio output failed");

    // sleep until playback is done - this should be improved so that the program is exited
    // when all samples have been played instead of estimating the duration and latency
    let latency = channel_sample_count as f64 / info.sample_rate as f64 * 2.0;
    std::thread::sleep(std::time::Duration::from_millis(((duration + latency) * 1000.0) as u64));
}

/// Simple conversion algorithm to convert Sample to f32 (integers are converted to [-1, 0.9921875]).
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
