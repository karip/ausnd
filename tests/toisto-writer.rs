//!
//! Test to write AU files based on input json spec file.
//!

use std::ffi::OsStr;
use std::fs::File;
use std::io::{Cursor, Error, ErrorKind, Read};
use std::path::Path;
use ausnd::{SampleFormat, AuError};
use serde::Deserialize;

#[test]
fn toisto_writer() {
    // ensure that toisto-au-test-suite folder is found
    assert!(Path::new("toisto-au-test-suite").try_exists()
        .expect("Check for toisto-au-test-suite failed"));

    let mut json_filenames = Vec::new();
    glob_json_files("toisto-au-test-suite/tests", &mut json_filenames)
        .expect("Can't get json filenames");
    run_test_for_files(&json_filenames, true, false);
}

#[path = "shared/jsonhelper.rs"]
mod jsonhelper;

fn ignored_tests() -> Vec<&'static str> {
    vec![
        "toisto-au-test-suite/tests/au/samplerate-0.json",
        "toisto-au-test-suite/tests/au/encoding-23-g721.json",
        "toisto-au-test-suite/tests/au/encoding-24-g722.json",
        "toisto-au-test-suite/tests/au/encoding-25-g723.3.json",
        "toisto-au-test-suite/tests/au/encoding-26-g723.5.json",
        "toisto-au-test-suite/tests/exported/audacity-g721-32kbs.json",
        "toisto-au-test-suite/tests/exported/audacity-g723.3-24kbs.json",
        "toisto-au-test-suite/tests/exported/audacity-g723.5-40kbs.json",
        "toisto-au-test-suite/tests/exported/audioconvert-g721.json",
        "toisto-au-test-suite/tests/exported/audioconvert-g723.3.json",
        "toisto-au-test-suite/tests/invalid/invalid-channels-0.json",
        "toisto-au-test-suite/tests/invalid/invalid-encoding.json",
    ]
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuJson {
    format: String,
    sample_rate: serde_json::Value,
    channels: u32,
    codec: String,
    sample_size: u32,
    samples_per_channel: usize,
    start_samples: Vec<Vec<serde_json::Value>>,
    end_samples: Vec<Vec<serde_json::Value>>,
    //desc: Option<Vec<u8>>, // desc isn't checked because padding bytes wouldn't match
}

impl AuJson {
    pub fn sample_rate(&self) -> f64 {
        serde_value_to_f64(&self.sample_rate)
    }
}

fn run_test_for_files(json_filenames: &[String], verbose: bool, no_errors: bool) {

    let ignored = ignored_tests();
    let mut count_ok = 0;
    let mut count_fail = 0;
    let mut count_ignore = 0;
    for json_filename in json_filenames {
        if ignored.contains(&json_filename.as_ref()) {
            count_ignore += 1;
            if verbose {
                println!("IGNORE: {}", json_filename);
            }
            continue;
        }
        match test(&json_filename) {
            Ok(()) => {
                count_ok += 1;
                if verbose {
                    println!("OK  : {}", json_filename);
                }
            },
            Err(e) => {
                count_fail += 1;
                if !no_errors {
                    println!("FAIL: {}", json_filename);
                    eprintln!(" * ERROR: {:?}", e);
                } else {
                    println!("(FAIL): {}", json_filename);
                    eprintln!(" * WARNING: {:?}", e);
                }
            }
        }
    }
    println!("Total write tests {}: {count_ok} passed, \
              {count_fail} failed, {count_ignore} ignored.",
        count_ok + count_fail + count_ignore);
    assert_eq!(count_fail, 0, "'left' number of tests failed:");
}

fn test(json_filename: &str) -> ausnd::AuResult<()> {
    let src_json = parse_json_file(&json_filename);
    let src_start_samples = convert_json_samples_to_f64(&src_json.start_samples);
    let src_end_samples = convert_json_samples_to_f64(&src_json.end_samples);

    let buf = match write_au(&src_json, &src_start_samples, &src_end_samples) {
        Err(e) => {
            return other_error(format!("Can't create AU file: {:?}", e));
        },
        Ok(b) => { b }
    };

    // open for reading

    let mut cursor = std::io::Cursor::new(&buf);
    let mut reader = ausnd::AuReader::new(&mut cursor)?;

    // create json

    let json = jsonhelper::jsonify(&mut reader).expect("Failed to jsonify");

    // compare source and generated jsons

    let gen_json: AuJson = parse_json(&json, "");
    if src_json.format != gen_json.format {
        return other_error(format!("format mismatch {:?} != {:?}",
            src_json.format, gen_json.format));
    }
    if src_json.sample_rate() != gen_json.sample_rate() {
        return other_error(format!("sample_rate mismatch {:?} != {:?}",
            src_json.sample_rate(), gen_json.sample_rate()));
    }
    if src_json.channels != gen_json.channels {
        return other_error(format!("channels mismatch {:?} != {:?}",
            src_json.channels, gen_json.channels));
    }
    if src_json.codec != gen_json.codec {
        return other_error(format!("codec mismatch {:?} != {:?}", src_json.codec, gen_json.codec));
    }
    if src_json.sample_size != gen_json.sample_size {
        if (src_json.sample_size as f64 / 8.0).ceil() !=
            (gen_json.sample_size as f64 / 8.0).ceil() {
            return other_error(format!("sample_size mismatch {:?} != {:?}",
                src_json.sample_size, gen_json.sample_size));
        }
    }
    if src_json.samples_per_channel != gen_json.samples_per_channel {
        return other_error(format!("samples_per_channel mismatch {:?} != {:?}",
            src_json.samples_per_channel, gen_json.samples_per_channel));
    }

    let gen_start_samples = convert_json_samples_to_f64(&src_json.start_samples);
    let gen_end_samples = convert_json_samples_to_f64(&src_json.end_samples);

    for ch in 0..src_start_samples.len() {
        for i in 0..src_start_samples[ch].len() {
            if !is_f64_equal(src_start_samples[ch][i], gen_start_samples[ch][i]) {
                return other_error(format!("start_samples mismatch {i}:{ch} {:?} != {:?}",
                    src_start_samples[ch][i], gen_start_samples[ch][i]));
            }
        }
    }
    for ch in 0..src_end_samples.len() {
        for i in 0..src_end_samples[ch].len() {
            if !is_f64_equal(src_end_samples[ch][i], gen_end_samples[ch][i]) {
                return other_error(format!("end_samples mismatch {i}:{ch} {:?} != {:?}",
                    src_end_samples[ch][i], gen_end_samples[ch][i]));
            }
        }
    }
    Ok(())
}

fn other_error(s: String) -> ausnd::AuResult<()> {
    return Err(AuError::from(Error::new(ErrorKind::Other, s)));
}

fn is_f64_equal(val1: f64, val2: f64) -> bool {
    (val1.is_nan() && val2.is_nan()) || (val1 == val2)
}

fn parse_json_file(filename: &str) -> AuJson {
    let mut file = File::open(filename).expect("Can't open json spec file");
    let mut txt = String::new();
    file.read_to_string(&mut txt).expect("Can't read json spec file");
    parse_json(&txt, filename)
}

fn parse_json(spectxt: &str, filename: &str) -> AuJson {
    match serde_json::from_str(spectxt) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(" * ERROR: invalid json file: {}: {}", filename, e);
            std::process::exit(2);
        }
    }
}

fn convert_json_samples_to_f64(samples: &Vec<Vec<serde_json::Value>>) -> Vec<Vec<f64>> {
    let mut result = vec![];
    for ch in samples {
        let mut out = vec![];
        for s in ch {
            out.push(serde_value_to_f64(s));
        }
        result.push(out);
    }
    result
}

fn serde_value_to_f64(val: &serde_json::Value) -> f64 {
    match val {
        serde_json::Value::String(s) => {
            if s == "nan" { f64::NAN }
            else if s == "inf" { f64::INFINITY }
            else if s == "-inf" { -f64::INFINITY }
            else { eprintln!(" * ERROR: invalid value: {}", s); std::process::exit(2); }
        },
        serde_json::Value::Number(n) => { n.as_f64().unwrap_or(0.0) },
        _ => { eprintln!(" * ERROR: invalid value: {:?}", val); std::process::exit(2); }
    }
}

fn write_au(json: &AuJson, src_start_samples: &Vec<Vec<f64>>, src_end_samples: &Vec<Vec<f64>>)
    -> ausnd::AuResult<Vec<u8>> {

    let sample_format = match (json.codec.as_str(), json.sample_size) {
        ("pcm_bei", 1..=8) => SampleFormat::I8,
        ("pcm_bei", 9..=16) => SampleFormat::I16,
        ("pcm_bei", 17..=24) => SampleFormat::I24,
        ("pcm_bei", 25..=32) => SampleFormat::I32,
        ("pcm_bef", 32) => SampleFormat::F32,
        ("pcm_bef", 64) => SampleFormat::F64,
        ("1", _) => SampleFormat::CompressedUlaw,
        ("27", _) => SampleFormat::CompressedAlaw,
        _ => {
            return Err(AuError::from(Error::new(ErrorKind::Other,
                format!("Unsupported sample format {:?}, sample size {:?}",
                json.codec.as_str(), json.sample_size))));
        }
    };
    let winfo = ausnd::AuWriteInfo {
        sample_rate: json.sample_rate() as u32,
        sample_format,
        channels: json.channels,
        ..ausnd::AuWriteInfo::default()
    };

    let mut buf = vec![];
    let mut cursor = Cursor::new(&mut buf);
    let mut auwriter = ausnd::AuWriter::new(&mut cursor, &winfo)?;
    let Ok(channels) = usize::try_from(json.channels) else {
        return Err(AuError::from(Error::new(ErrorKind::Other, "Too many channels")));
    };
    if json.samples_per_channel.checked_mul(channels).is_none() {
        return Err(AuError::from(Error::new(ErrorKind::Other, "Too many channels")));
    }
    match sample_format {
        SampleFormat::I8 => {
            let samples = create_samples::<i8>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i8 });
            auwriter.write_samples_i8(&samples)?;
        },
        SampleFormat::I16 => {
            let samples = create_samples::<i16>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i16 });
            auwriter.write_samples_i16(&samples)?;
        },
        SampleFormat::I24 => {
            let samples = create_samples::<i32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i32 });
            auwriter.write_samples_i24(&samples)?;
        },
        SampleFormat::I32 => {
            let samples = create_samples::<i32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i32 });
            auwriter.write_samples_i32(&samples)?;
        },
        SampleFormat::F32 => {
            let samples = create_samples::<f32>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as f32 });
            auwriter.write_samples_f32(&samples)?;
        },
        SampleFormat::F64 => {
            let samples = create_samples::<f64>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as f64 });
            auwriter.write_samples_f64(&samples)?;
        },
        SampleFormat::CompressedUlaw => {
            let samples = create_samples::<i16>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i16 });
            auwriter.write_samples_i16(&samples)?;
        },
        SampleFormat::CompressedAlaw => {
            let samples = create_samples::<i16>(json.samples_per_channel,
                channels, src_start_samples, src_end_samples,
                |x| { x as i16 });
            auwriter.write_samples_i16(&samples)?;
        },
        SampleFormat::Custom(_) => {
            let samples = vec![0u8; json.samples_per_channel * channels];
            auwriter.write_samples_raw(&samples)?;
        },
    }
    auwriter.finalize()?;
    Ok(buf)
}

fn create_samples<T>(samples_per_channel: usize, channels: usize,
        start_samples: &Vec<Vec<f64>>, end_samples: &Vec<Vec<f64>>, converter: impl Fn(f64) -> T)
         -> Vec<T> where T: Default + Clone {
    let mut samples = vec![T::default(); samples_per_channel*channels];
    if samples_per_channel == 0 || channels == 0 {
        return samples;
    }
    for ch in 0..channels {
        for i in 0..start_samples[ch].len() {
            samples[i*channels + ch] = converter(start_samples[ch][i]);
        }
        for i in 0..end_samples[ch].len() {
            let pos = (samples_per_channel - end_samples[ch].len() + i)*channels + ch;
            samples[pos] = converter(end_samples[ch][i]);
        }
    }
    samples
}

fn glob_json_files(folder: impl AsRef<Path>, jsons: &mut Vec<String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        if entry.path().is_dir() {
            glob_json_files(entry.path(), jsons)?;

        } else if entry.path().extension() == Some(OsStr::new("json")) {
            jsons.push(entry.path().to_string_lossy().to_string());
        }
    }
    Ok(())
}
