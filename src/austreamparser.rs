
use crate::{cast, AuReadInfo, Sample, SampleFormat, Write};

/// AuEvents sent by [`AuStreamParser`].
#[derive(Debug, Clone, PartialEq)]
pub enum AuEvent<'a> {
    /// Header event. Only one Header event is sent in the beginning.
    Header(AuReadInfo),
    /// Description data event. Zero or more `DescData` events may be sent.
    DescData(&'a [u8]),
    /// Sample data event. Zero or more `SampleData` events may be sent.
    SampleData(&'a [Sample]),
    /// End event. No more events are sent after this event.
    End
}

/// Streamer state.
#[derive(Debug)]
enum StreamState {
    Header,
    Desc,
    Samples,
    End
}

/// AU stream parser.
///
/// `AuStreamParser` can be used to process a stream of bytes as an AU audio stream.
/// It parses data written to it and dispatches header and sample events to a callback.
///
/// If the header has a known sample length, then samples are dispatched
/// until all samples have been read. If the header indicates
/// that the length is unknown, then samples are dispatched until the end of the stream.
/// In that case, the stream may contain an infinite number of samples if the stream never ends.
///
/// # Errors
///
/// If any of the methods returns an error, then the reader is in an undefined state and
/// shouldn't be used anymore.
///
/// # Examples
///
/// ```no_run
/// # use std::io::Write;
/// # fn example() -> ausnd::AuResult<()> {
/// let mut streamer = ausnd::AuStreamParser::new(|event| {
///     println!("Got event: {:?}", event);
/// });
/// let byte_data = &[ 0x2e, 0x73, 0x6e, 0x64, 0x00 ];
/// streamer.write(byte_data)?;
/// // ..more write calls to parse more data..
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AuStreamParser<F> {
    callback: F,
    info: AuReadInfo,
    data_offset: u64,
    data_size: Option<u64>,
    state: StreamState,
    buffer: [u8; 24],
    header_buf_pos: usize,
    sample_buf_pos: usize,
    desc_pos: u64,
    sample_byte_pos: u64,
}

impl<F> AuStreamParser<F> where F: FnMut(AuEvent) {
    /// Creates a new `AuStreamParser` with the given callback. The callback will receive
    /// [`AuEvent`]s when data is written to the streamer.
    pub fn new(callback: F) -> AuStreamParser<F> {
        let info = AuReadInfo {
            channels: 2,
            sample_rate: 44100,
            sample_format: SampleFormat::I16,
            description_byte_len: 0,
            sample_len: None,
            sample_byte_len: None
        };
        AuStreamParser {
            callback,
            info,
            data_offset: 0,
            data_size: None,
            state: StreamState::Header,
            buffer: [0u8; 24],
            header_buf_pos: 0,
            sample_buf_pos: 0,
            desc_pos: 24,
            sample_byte_pos: 0,
        }
    }

    /// Dispatches the [`AuEvent::End`] event to the callback. After this, no other events are
    /// dispatched even if `write()` or `finalize()` are called again.
    pub fn finalize(&mut self) {
        if let StreamState::End = self.state {
            return;
        }
        (self.callback)(AuEvent::End);
        self.state = StreamState::End;
    }
}

impl<F> Write for AuStreamParser<F> where F: FnMut(AuEvent) {
    /// Appends data to the streamer. This dispatches events to the `AuStreamParser`'s callback.
    /// Returns the InvalidInput error for invalid header data and unsupported sample encoding.
    fn write(&mut self, data: &[u8]) -> Result<usize, std::io::Error> {
        for b in data {
            match self.state {
                StreamState::Header => {
                    self.buffer[self.header_buf_pos] = *b;
                    self.header_buf_pos += 1;
                    if self.header_buf_pos == 24 {
                        // dispatch header
                        let h = AuReadInfo::parse_bytes(&self.buffer)
                            .map_err(|_| {
                                std::io::Error::from(std::io::ErrorKind::InvalidInput)
                            })?;
                        self.data_offset = h.0;
                        self.data_size = h.1.sample_byte_len.map(u64::from);
                        self.info = h.1.clone();
                        (self.callback)(AuEvent::Header(h.1));
                        if self.data_offset > 24 {
                            self.state = StreamState::Desc;
                        } else if self.data_size.unwrap_or(u64::MAX) > 0 {
                            self.state = StreamState::Samples;
                        } else {
                            (self.callback)(AuEvent::End);
                            self.state = StreamState::End;
                        }
                    }
                },
                StreamState::Desc => {
                    self.desc_pos += 1;
                    // dispatch desc data
                    (self.callback)(AuEvent::DescData(&[ *b ]));
                    if self.desc_pos == self.data_offset {
                        if self.data_size.unwrap_or(u64::MAX) > 0 {
                            self.state = StreamState::Samples;
                        } else {
                            (self.callback)(AuEvent::End);
                            self.state = StreamState::End;
                        }
                    }
                },
                StreamState::Samples => {
                    self.buffer[self.sample_buf_pos] = *b;
                    self.sample_buf_pos += 1;
                    if self.sample_buf_pos == usize::from(self.info.sample_format.bytesize_u8()) {
                        let s = match self.info.sample_format {
                            SampleFormat::I8 => { parse_sample_i8(&self.buffer) },
                            SampleFormat::I16 => { parse_sample_i16(&self.buffer) },
                            SampleFormat::I24 => { parse_sample_i24(&self.buffer) },
                            SampleFormat::I32 => { parse_sample_i32(&self.buffer) },
                            SampleFormat::F32 => { parse_sample_f32(&self.buffer) },
                            SampleFormat::F64 => { parse_sample_f64(&self.buffer) },
                            SampleFormat::CompressedUlaw => { Sample::I16(
                                audio_codec_algorithms::decode_ulaw(self.buffer[0]))
                            },
                            SampleFormat::CompressedAlaw => { Sample::I16(
                                audio_codec_algorithms::decode_alaw(self.buffer[0]))
                            },
                            SampleFormat::Custom(_) => {
                                return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
                            },
                        };
                        // dispatch sample
                        (self.callback)(AuEvent::SampleData(&[ s ]));
                        self.sample_buf_pos = 0;
                    }
                    // dispatch end if at the end
                    self.sample_byte_pos += 1;
                    if let Some(ds) = self.data_size {
                        if self.sample_byte_pos == ds {
                            (self.callback)(AuEvent::End);
                            self.state = StreamState::End;
                        }
                    }
                },
                StreamState::End => {
                    // just ignore all data
                }
            }
        }
        Ok(data.len())
    }

    /// Flush does nothing.
    fn flush(&mut self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

fn parse_sample_i8(data: &[u8; 24]) -> Sample {
    Sample::I8(cast::u8_as_i8(data[0]))
}

fn parse_sample_i16(data: &[u8; 24]) -> Sample {
    let mut buf = [0u8; 2];
    buf.copy_from_slice(&data[0..2]);
    Sample::I16(i16::from_be_bytes(buf))
}

fn parse_sample_i24(data: &[u8; 24]) -> Sample {
    let mut res = i32::from(data[0]) << 16 | i32::from(data[1]) << 8 | i32::from(data[2]);
    if res >= 8388608 {
        res -= 16777216;
    }
    Sample::I24(res)
}

fn parse_sample_i32(data: &[u8; 24]) -> Sample {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[0..4]);
    Sample::I32(i32::from_be_bytes(buf))
}

fn parse_sample_f32(data: &[u8; 24]) -> Sample {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[0..4]);
    Sample::F32(f32::from_be_bytes(buf))
}

fn parse_sample_f64(data: &[u8; 24]) -> Sample {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[0..8]);
    Sample::F64(f64::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use crate::AuResult;
    use super::*;

    #[test]
    fn test_parse_sample_i24() -> AuResult<()> {
        assert_eq!(parse_sample_i24(&[1, 2, 3,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]), Sample::I24(66051));
        assert_eq!(parse_sample_i24(&[128, 2, 3,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]), Sample::I24(-8388093));
        Ok(())
    }

    #[test]
    fn test_known_size() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I8,
                description_byte_len: 4, sample_len: Some(4), sample_byte_len: Some(4)
            }),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::SampleData(&[ Sample::I8(1) ]),
            AuEvent::SampleData(&[ Sample::I8(2) ]),
            AuEvent::SampleData(&[ Sample::I8(3) ]),
            AuEvent::SampleData(&[ Sample::I8(4) ]),
            AuEvent::End
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,28,  0,0,0,4 ])?;
        streamer.write(&[ 0,0,0,2,  0,0,172,68,  0,0,0, ])?;
        streamer.write(&[ 1,  0,0 ])?;              // channels + start of desc
        streamer.write(&[ 0 ])?;                    // desc
        streamer.write(&[ 0, 1, 2, 3 ])?;           // desc + samples
        streamer.write(&[ 4, 0xfd, 0xfe, 0xff, ])?; // samples + some garbage data
        assert!(expected_events.is_empty());
        Ok(())
    }

    #[test]
    fn test_unknown_size() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I8,
                description_byte_len: 4, sample_len: None, sample_byte_len: None
            }),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::SampleData(&[ Sample::I8(1) ]),
            AuEvent::SampleData(&[ Sample::I8(2) ]),
            AuEvent::SampleData(&[ Sample::I8(3) ]),
            AuEvent::SampleData(&[ Sample::I8(4) ]),
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,28,  0xff,0xff,0xff,0xff ])?;
        streamer.write(&[ 0,0,0,2,  0,0,172,68,  0,0,0, ])?;
        streamer.write(&[ 1,  0,0 ])?;    // channels + start of desc
        streamer.write(&[ 0 ])?;          // desc
        streamer.write(&[ 0, 1, 2 ])?;    // desc + samples
        streamer.write(&[ 3, 4 ])?;       // samples
        assert!(expected_events.is_empty());
        Ok(())
    }

    #[test]
    fn test_no_desc() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I16,
                description_byte_len: 0, sample_len: Some(4), sample_byte_len: Some(8)
            }),
            AuEvent::SampleData(&[ Sample::I16(1) ]),
            AuEvent::SampleData(&[ Sample::I16(2) ]),
            AuEvent::SampleData(&[ Sample::I16(3) ]),
            AuEvent::SampleData(&[ Sample::I16(4) ]),
            AuEvent::End
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,24,  0,0,0,8 ])?;
        streamer.write(&[ 0,0,0,3,  0,0,172,68 ])?;
        streamer.write(&[ 0,0,0,1 ])?;              // channels
        streamer.write(&[ 0, 1, 0, 2, 0, 3 ])?; // samples
        streamer.write(&[ 0, 4, 0xfd, 0xfe, 0xff, ])?; // samples + some garbage data
        assert!(expected_events.is_empty());
        Ok(())
    }

    #[test]
    fn test_no_samples() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I8,
                description_byte_len: 4, sample_len: Some(0), sample_byte_len: Some(0)
            }),
            AuEvent::DescData(&[ 65 ]),
            AuEvent::DescData(&[ 66 ]),
            AuEvent::DescData(&[ 67 ]),
            AuEvent::DescData(&[ 68 ]),
            AuEvent::End
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,28,  0,0,0,0 ])?;
        streamer.write(&[ 0,0,0,2,  0,0,172,68,  0,0,0, ])?;
        streamer.write(&[ 1,  65,66 ])?;              // channels + start of desc
        streamer.write(&[ 67 ])?;                    // desc
        streamer.write(&[ 68, 0xfd, 0xfe, 0xff, ])?; // desc + some garbage data
        assert!(expected_events.is_empty());
        Ok(())
    }

    #[test]
    fn test_no_desc_or_samples() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I8,
                description_byte_len: 0, sample_len: Some(0), sample_byte_len: Some(0)
            }),
            AuEvent::End
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,24,  0,0,0,0 ])?;
        streamer.write(&[ 0,0,0,2,  0,0,172,68, 0,0,0,1 ])?;
        streamer.write(&[ 0xfd, 0xfe, 0xff, ])?;    // some garbage data
        assert!(expected_events.is_empty());
        Ok(())
    }

    #[test]
    fn test_finalize() -> AuResult<()> {
        let mut expected_events = vec![
            AuEvent::Header(AuReadInfo {
                channels: 1, sample_rate: 44100, sample_format: SampleFormat::I8,
                description_byte_len: 4, sample_len: None, sample_byte_len: None
            }),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::DescData(&[ 0 ]),
            AuEvent::SampleData(&[ Sample::I8(1) ]),
            AuEvent::SampleData(&[ Sample::I8(2) ]),
            AuEvent::SampleData(&[ Sample::I8(3) ]),
            AuEvent::SampleData(&[ Sample::I8(4) ]),
            AuEvent::SampleData(&[ Sample::I8(5) ]),
            AuEvent::End
        ];
        let mut streamer = AuStreamParser::new(|ev| {
            assert_eq!(ev, expected_events.remove(0));
        });
        streamer.write(&[ b'.', b's', b'n', b'd', 0,0,0,28,  0xff,0xff,0xff,0xff ])?;
        streamer.write(&[ 0,0,0,2,  0,0,172,68,  0,0,0, ])?;
        streamer.write(&[ 1,  0,0 ])?;              // channels + start of desc
        streamer.write(&[ 0 ])?;                    // desc
        streamer.write(&[ 0, 1, 2, 3, 4 ])?;        // desc + samples
        streamer.write(&[ 5 ])?;                    // sample
        streamer.finalize();
        streamer.write(&[ 5, 0xfd, 0xfe, 0xff, ])?; // some garbage data ignored
        streamer.finalize();                            // finalize ignored
        assert!(expected_events.is_empty());
        Ok(())
    }
}
