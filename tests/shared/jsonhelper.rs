// Helper for testing.

use ausnd::{SampleFormat, AuError};

/// Reads data from the given AuReader and returns it as JSON.
pub fn jsonify<T>(reader: &mut ausnd::AuReader<T>) -> ausnd::AuResult<String>
    where T: std::io::Read+std::io::Seek {

    let mut json = String::new();

    let info = match reader.read_info() {
        Err(AuError::StdIoError(e)) => {
            return Ok(format!("{{ \"error\": {:?} }}", e.to_string()));
        },
        Err(error) => {
            return Ok(format!("{{ \"error\": \"{:?}\" }}", error));
        },
        Ok(val) => val
    };

    json += &format!("{{\n");
    json += &format!("    \"format\": \"au\",\n");
    json += &format!("    \"sampleRate\": {},\n", info.sample_rate);
    json += &format!("    \"channels\": {},\n", info.channels);
    json += &format!("    \"codec\": {},\n", sampleformat_to_codec(&info.sample_format));
    json += &format!("    \"sampleSize\": {},\n", info.sample_format.decoded_size() * 8);

    let mut desc = vec![];
    let mut buf = [0u8; 1];
    while reader.read_description(&mut buf).expect("Can't read description") != 0 {
        desc.extend_from_slice(&buf);
    }
    json += &format!("    \"desc\": {},\n", bytes_to_json_number_array(&desc));

    let mut samples = vec![];
    let mut sample_error = read_samples(reader, &mut samples);

    let mut sample_frames = 0;
    if info.channels > 0 {
        let chs = u64::from(info.channels);
        if let Ok(slen) = reader.resolved_sample_len() {
            sample_frames = slen / chs;
        } else {
            // try to get sample count from samples result
            sample_frames = u64::try_from(samples.len()).unwrap_or(u64::MAX) / chs;
        }
    }
    json += &format!("    \"samplesPerChannel\": {}", sample_frames);

    //println!("{:?}", info);

    // ensure 300 samples for each channel can be read
    let channels = match usize::try_from(info.channels) {
        Ok(ch) => ch,
        Err(_) =>  {
            sample_error = Err(AuError::InvalidReadState);
            0
        }
    };
    if sample_error.is_ok() && channels.checked_mul(300).is_none() {
        sample_error = Err(AuError::InvalidReadState);
    }
    match sample_error {
        Ok(()) => {
            // print sample data
            json += &format!(",\n    \"startSamples\": [\n");
            let start_einx = samples.len().min(300*channels);
            print_sample_data(channels, &samples[0..start_einx], &mut json);
            json += &format!("    ],\n");

            json += &format!("    \"endSamples\": [\n");
            let mut end_sinx = 0;
            if samples.len() > 30*channels {
                end_sinx = samples.len() - 30*channels;
            }
            print_sample_data(channels, &samples[end_sinx..], &mut json);
            json += &format!("    ]\n");
        },
        Err(AuError::Unsupported) => {
            json += &format!(",\n    \"startSamples\": \"-unsupported-\",\n");
            json += &format!("    \"endSamples\": \"-unsupported-\"\n");
        },
        Err(AuError::StdIoError(e)) => {
            json += &format!(",\n    \"error\": {:?}\n", e.to_string());
        }
        Err(e) => {
            json += &format!(",\n    \"error\": \"{:?}\"\n", e);
        }
    }
    json += &format!("}}\n");
    Ok(json)
}

fn read_samples<T>(reader: &mut ausnd::AuReader<T>, samples: &mut Vec<f64>) -> ausnd::AuResult<()>
    where T: std::io::Read + std::io::Seek {
    for sample in reader.samples()? {
        match sample? {
            ausnd::Sample::I8(s) => { samples.push(s as f64); },
            ausnd::Sample::I16(s) => { samples.push(s as f64); },
            ausnd::Sample::I24(s) => { samples.push(s as f64); },
            ausnd::Sample::I32(s) => { samples.push(s as f64); },
            ausnd::Sample::F32(s) => { samples.push(s as f64); },
            ausnd::Sample::F64(s) => { samples.push(s as f64); }
        }
    }
    Ok(())
}

/// Converts bytes to JSON array: [ numbers.. ]
fn bytes_to_json_number_array(data: &[u8]) -> String {
    let list = data.iter()
        .map(|b| format!("{}", b))
        .collect::<Vec<String>>()
        .join(", ");
    format!("[ {} ]", list)
}

fn sampleformat_to_codec(sample_format: &SampleFormat) -> String {
    match sample_format {
        ausnd::SampleFormat::I8 => { "\"pcm_bei\"".to_string() },
        ausnd::SampleFormat::I16 => { "\"pcm_bei\"".to_string() },
        ausnd::SampleFormat::I24 => { "\"pcm_bei\"".to_string() },
        ausnd::SampleFormat::I32 => { "\"pcm_bei\"".to_string() },
        ausnd::SampleFormat::F32 => { "\"pcm_bef\"".to_string() },
        ausnd::SampleFormat::F64 => { "\"pcm_bef\"".to_string() },
        ausnd::SampleFormat::CompressedUlaw => { "\"1\"".to_string() },
        ausnd::SampleFormat::CompressedAlaw => { "\"27\"".to_string() },
        ausnd::SampleFormat::Custom(chid) => { format!("\"{}\"", chid.to_string()) },
    }
}

fn print_sample_data(channels: usize, samples: &[f64], json: &mut String) {
    if channels == 0 || channels > 256 {
        return;
    }
    let samples_per_channel = samples.len() / channels;
    for ch in 0..channels {
        *json += "        [ ";
        let mut pos = 0;
        while pos < samples_per_channel {
            if pos != 0 {
                *json += &format!(", ");
            }
            let s = samples[pos * channels + ch];
            if s.is_finite() {
                *json += &format!("{:.6}", s);
            } else {
                let str = format!("\"{:.6}\"", s);
                *json += &format!("{}", str.to_lowercase());
            }
            pos += 1;
        }
        if ch < channels-1 {
            *json += &format!(" ],\n");
        } else {
            *json += &format!(" ]\n");
        }
    }
}
