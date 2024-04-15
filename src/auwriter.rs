
use crate::{cast, AuError, AuWriteInfo, AuResult, SampleFormat, Seek, SeekFrom, Write};

#[derive(Debug, Clone, Copy, PartialEq)]
enum WriteState {
    SamplesNotWritten,
    WritingSamples,
    WritingRawSamples,
    Finalized
}

/// AU audio format writer.
///
/// `AuWriter` writes the AU audio format.
/// When writing audio samples, channel data is interleaved.
///
/// After all samples have been written, [`finalize()`](AuWriter::finalize()) can be called to
/// update the header to store the size of the sample data.
/// Calling `finalize()` is optional, but recommended so that
/// reading the resulting audio data can be optimized.
///
/// It is possible to create `AuWriter` for a writer implementing only `Write`, but
/// a writer implementing `Write + Seek` is needed to call `finalize()`.
///
/// The writer can write an unlimited number of audio samples.
///
/// The writer doesn't perform any buffering, so it's recommended to use a buffered writer with it.
///
/// # Errors
///
/// If any of the methods returns an error, then the writer shouldn't be used anymore.
///
/// # Examples
///
/// Writing an AU file with 2 channels, sample rate 48000 and signed 16-bit integer samples.
/// In the end, `finalize()` is called to update the file header.
/// ```no_run
/// # fn example() -> ausnd::AuResult<()> {
/// let mut bufwr = std::io::BufWriter::new(std::fs::File::create("test.au")?);
/// let winfo = ausnd::AuWriteInfo {
///     channels: 2,
///     sample_rate: 48000,
///     sample_format: ausnd::SampleFormat::I16,
/// };
/// let mut writer = ausnd::AuWriter::new(&mut bufwr, &winfo)?;
/// writer.write_samples_i16(&[ 0, 0, 10, 10, 20, 20, 30, 30, 40, 40 ])?;
/// writer.finalize()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AuWriter<W> {

    /// Header.
    info: AuWriteInfo,

    /// Writer state.
    state: WriteState,

    /// Header size (24 + desc size) in bytes.
    header_size_in_bytes: u32,

    /// Sample byte count written to the stream.
    sample_bytes_written: u64,

    /// Underlying writer.
    handle: W,
}

impl<W: Write> AuWriter<W> {
    /// Creates a new `AuWriter`.
    ///
    /// The header data is written immediately to the inner writer.
    pub fn new(inner: W, info: &AuWriteInfo) -> AuResult<AuWriter<W>> {
        AuWriter::new_with_desc(inner, info, &[])
    }

    /// Creates a new `AuWriter`, which writes additional description data after the header.
    /// The `desc` data end is padded with zeros so that the subsequent sample data is
    /// always 4-byte aligned. An empty `desc` is always written as 4 zero bytes.
    ///
    /// The header and description data are written immediately to the inner writer.
    pub fn new_with_desc(mut inner: W, info: &AuWriteInfo, desc: &[u8]) -> AuResult<AuWriter<W>> {
        let header_size_in_bytes = AuWriter::write_header(&mut inner, info, desc)?;
        Ok(AuWriter {
            info: info.clone(),
            state: WriteState::SamplesNotWritten,
            header_size_in_bytes,
            sample_bytes_written: 0,
            handle: inner,
        })
    }

    /// Flushes the stream. Note: this only flushes the stream. This doesn't update the header.
    pub fn flush(&mut self) -> AuResult<()> {
        Ok(self.handle.flush()?)
    }

    fn increment_sample_bytes_written(&mut self, sample_len: usize, sample_format: SampleFormat) {
        let dlen = u64::try_from(sample_len).unwrap_or(u64::MAX);
        let bytes = dlen.saturating_mul(sample_format.bytesize_u64());
        self.sample_bytes_written = self.sample_bytes_written.saturating_add(bytes);
    }

    /// Writes `i8` samples. Call this method if `AuWriteInfo::sample_format` is set to
    /// `SampleFormat::I8`.
    pub fn write_samples_i8(&mut self, data: &[i8]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::I8 {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        for d in data {
            self.handle.write_all(&[ cast::i8_as_u8(*d) ])?;
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes `i16` samples. Call this method if `AuWriteInfo::sample_format` is set to
    /// `SampleFormat::I16`, `CompressedUlaw` or `CompressedAlaw`.
    pub fn write_samples_i16(&mut self, data: &[i16]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::I16 &&
            self.info.sample_format != SampleFormat::CompressedUlaw &&
            self.info.sample_format != SampleFormat::CompressedAlaw {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        if self.info.sample_format == SampleFormat::I16 {
            for d in data {
                self.handle.write_all(&d.to_be_bytes())?;
            }

        } else if self.info.sample_format == SampleFormat::CompressedUlaw {
            for d in data {
                let encoded = audio_codec_algorithms::encode_ulaw(*d);
                self.handle.write_all(&[ encoded ])?;
            }

        } else if self.info.sample_format == SampleFormat::CompressedAlaw {
            for d in data {
                let encoded = audio_codec_algorithms::encode_alaw(*d);
                self.handle.write_all(&[ encoded ])?;
            }
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes `i24` samples. The `i32` values should be in the range [-8388608, 8388607], because
    /// they are converted to 24-bit values by discarding their top-most 8 bits.
    /// Call this method if `AuWriteInfo::sample_format` is set to `SampleFormat::I24`.
    pub fn write_samples_i24(&mut self, data: &[i32]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::I24 {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        for d in data {
            let i32bytes = &d.to_be_bytes();
            let i24bytes = [ i32bytes[1], i32bytes[2], i32bytes[3] ];
            self.handle.write_all(&i24bytes)?;
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes `i32` samples.
    /// Call this method if `AuWriteInfo::sample_format` is set to `SampleFormat::I32`.
    pub fn write_samples_i32(&mut self, data: &[i32]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::I32 {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        for d in data {
            self.handle.write_all(&d.to_be_bytes())?;
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes `f32` samples.
    /// Call this method if `AuWriteInfo::sample_format` is set to `SampleFormat::F32`.
    pub fn write_samples_f32(&mut self, data: &[f32]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::F32 {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        for d in data {
            self.handle.write_all(&d.to_be_bytes())?;
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes `f64` samples.
    /// Call this method if `AuWriteInfo::sample_format` is set to `SampleFormat::F64`.
    pub fn write_samples_f64(&mut self, data: &[f64]) -> AuResult<()> {
        if self.info.sample_format != SampleFormat::F64 {
            return Err(AuError::InvalidSampleFormat);
        }
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingSamples;
        for d in data {
            self.handle.write_all(&d.to_be_bytes())?;
        }
        self.increment_sample_bytes_written(data.len(), self.info.sample_format);
        Ok(())
    }

    /// Writes raw sample data. Raw sample data can be written for any sample format.
    /// If this method is called once, then the other write sample methods can't be called anymore.
    /// This method is the only way to write sample data for custom encodings
    /// (`SampleFormat::Custom`).
    pub fn write_samples_raw(&mut self, data: &[u8]) -> AuResult<()> {
        if self.state != WriteState::SamplesNotWritten &&
            self.state != WriteState::WritingRawSamples {
            return Err(AuError::InvalidWriteState);
        }
        self.state = WriteState::WritingRawSamples;
        self.handle.write_all(data)?;
        self.sample_bytes_written = self.sample_bytes_written
            .saturating_add(u64::try_from(data.len()).unwrap_or(u64::MAX));
        Ok(())
    }

    /// Consumes this `AuWriter` and returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.handle
    }

    /// Gets a reference to the underlying writer.
    pub const fn get_ref(&self) -> &W {
        &self.handle
    }

    /// Gets a mutable reference to the underlying writer.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying writer as it may corrupt this writer's state.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.handle
    }

    /// Writes the AU header and returns the header size (24+desc.len()) in bytes.
    fn write_header(write: &mut W, info: &AuWriteInfo, desc: &[u8]) -> AuResult<u32> {
        if info.channels < 1 {
            return Err(AuError::InvalidParameter);
        }
        if info.sample_rate == 0 {
            return Err(AuError::InvalidParameter);
        }
        let Ok(desc_len) = u32::try_from(desc.len()) else {
            return Err(AuError::InvalidParameter);
        };
        let mut desc_padding_len = desc_len % 4;
        if desc_len == 0 || desc_padding_len > 0 {
            desc_padding_len = 4 - desc_padding_len;
        }
        let Some(offset) = desc_len
            .checked_add(24)
            .and_then(|dl| dl.checked_add(desc_padding_len)) else {
            return Err(AuError::InvalidParameter);
        };
        write.write_all(&[ b'.', b's', b'n', b'd' ])?;          // magic
        write.write_all(&offset.to_be_bytes())?;                // offset
        write.write_all(&[ 0xff, 0xff, 0xff, 0xff ])?;          // data_size = initially unknown
        write.write_all(&(info.sample_format.as_u32()).to_be_bytes())?; // encoding
        write.write_all(&info.sample_rate.to_be_bytes())?;      // sample rate
        write.write_all(&info.channels.to_be_bytes())?;         // channels
        write.write_all(desc)?;                                 // description (min. 4 bytes)
        for _ in 0..desc_padding_len {
            write.write_all(&[ 0 ])?;
        }
        Ok(offset)
    }
}

impl<W: Write + Seek> AuWriter<W> {
    /// Updates the sample count to the header.
    /// Call this method after all sample data has been written.
    /// The write methods must not be called after this method has been called.
    ///
    /// Returns an error if updating the header fails.
    pub fn finalize(&mut self) -> AuResult<()> {
        if self.state == WriteState::Finalized {
            return Ok(());
        }
        let spos = self.handle.stream_position()?;
        // update data_size if it fits in u32
        let Ok(size_u32) = u32::try_from(self.sample_bytes_written) else {
            return Ok(());
        };
        let data_size_pos = u64::from(self.header_size_in_bytes) + u64::from(size_u32) - 8;
        let Ok(seek_offset) = i64::try_from(data_size_pos) else {
            return Ok(());
        };
        self.handle.seek(SeekFrom::Current(-seek_offset))?;
        self.handle.write_all(&size_u32.to_be_bytes())?;
        self.handle.seek(SeekFrom::Start(spos))?;
        self.handle.flush()?;
        self.state = WriteState::Finalized;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn test_new_transferring_ownership() -> AuResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(cursor, &AuWriteInfo::default())?;
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,0,  0,0,0,3,
            0,0,172,68,  0,0,0,2,  0,0,0,0 ]);
        Ok(())
    }

    #[test]
    fn test_new_taking_mut_ref() -> AuResult<()> {
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &AuWriteInfo::default())?;
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,0,  0,0,0,3,
            0,0,172,68,  0,0,0,2,  0,0,0,0 ]);
        Ok(())
    }

    #[test]
    fn test_new_ref_slice() -> AuResult<()> {
        let mut output = [0u8; 32];
        let mut cursor: &mut [u8] = output.as_mut();
        let mut writer = AuWriter::new(&mut cursor, &AuWriteInfo::default())?;
        writer.write_samples_i16(&[ 11, 12 ])?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0xff,0xff,0xff,0xff,  0,0,0,3,
            0,0,172,68,  0,0,0,2,  0,0,0,0,  0, 11, 0, 12 ]);
        Ok(())
    }

    #[test]
    fn test_into_inner() -> AuResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let writer = AuWriter::new(cursor, &AuWriteInfo::default())?;
        let mut w = writer.into_inner();
        w.write(&[ 0xff ])?;
        assert_eq!(w.position(), 29);
        Ok(())
    }

    #[test]
    fn test_get_ref() -> AuResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(cursor, &AuWriteInfo::default())?;
        let w = writer.get_ref();
        assert_eq!(w.position(), 28);
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,0,  0,0,0,3,
            0,0,172,68,  0,0,0,2,  0,0,0,0 ]);
        Ok(())
    }

    #[test]
    fn test_get_mut() -> AuResult<()> {
        let mut output = vec![];
        let cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(cursor, &AuWriteInfo::default())?;
        let w: &mut Cursor<&mut Vec<u8>> = writer.get_mut();
        w.write(&[ 88 ])?;
        assert_eq!(w.position(), 29);
        writer.write_samples_i16(&[ 123 ])?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0xff,0xff,0xff,0xff,  0,0,0,3,
            0,0,172,68,  0,0,0,2,  0,0,0,0,  88, 0, 123 ]);
        Ok(())
    }

    #[test]
    fn test_custom_encoding() -> AuResult<()> {
        let mut output = vec![];
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::Custom(1025),
            channels: 1,
        };
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        // custom encoding requires writing raw data
        assert!(writer.write_samples_i16(&[ 10 ]).is_err());
        writer.write_samples_raw(&[ 11, 12, 13 ])?;
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,3,  0,0,4,1,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  11, 12, 13 ]);
        Ok(())
    }

    fn write_desc_using_slice(sample_rate: u32, channels: u32, format: SampleFormat,
        desc: &[u8], raw_data: &[u8]) -> AuResult<Vec<u8>> {
        let winfo = AuWriteInfo {
            sample_rate,
            sample_format: format,
            channels,
        };
        let mut output = vec![0u8; 10000];
        let mut output_size = output.len();
        // use mutable slice as Write to test AuWriter without Seek
        let mut cursor: &mut [u8] = output.as_mut();
        let mut writer = AuWriter::new_with_desc(&mut cursor, &winfo, desc)?;
        writer.write_samples_raw(raw_data)?;
        output_size -= cursor.len();
        // resize vec to its correct size
        output.resize(output_size, 0);
        Ok(output)
    }

    fn write_desc(sample_rate: u32, channels: u32, format: SampleFormat,
        desc: &[u8], raw_data: &[u8]) -> AuResult<Vec<u8>> {
        let winfo = AuWriteInfo {
            sample_rate,
            sample_format: format,
            channels,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new_with_desc(&mut cursor, &winfo, desc)?;
        writer.write_samples_raw(raw_data)?;
        writer.finalize()?;
        Ok(output)
    }

    #[test]
    fn test_write_description() -> AuResult<()> {
        //  no padding bytes, only Read trait
        let output = write_desc_using_slice(44100, 1, SampleFormat::I8,
            &[ b'C', b'u', b't', b'e' ], &[ 11, 12, 13, 14 ])?;
        assert_eq!(output, &[ b'.', b's', b'n', b'd',  0,0,0,28,  255,255,255,255,  0,0,0,2,
            0,0,172,68,  0,0,0,1,
            b'C', b'u', b't', b'e',
            11, 12, 13, 14 ]);

        //  no padding bytes, Read + Seek
        let output = write_desc(44100, 1, SampleFormat::I8,
            &[ b'C', b'u', b't', b'e' ], &[ 11, 12, 13, 14 ])?;
        assert_eq!(output, &[ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,
            b'C', b'u', b't', b'e',
            11, 12, 13, 14 ]);

        // 1 desc padding bytes
        let output = write_desc(44100, 1, SampleFormat::I8,
                &[ b'p', b'e', b'r', b'f', b'e', b'c', b't' ], &[ 11, 12, 13, 14 ])?;
        assert_eq!(output, &[ b'.', b's', b'n', b'd',  0,0,0,32,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,
            b'p', b'e', b'r', b'f', b'e', b'c', b't', 0,
            11, 12, 13, 14 ]);

        // 2 desc padding bytes
        let output = write_desc(44100, 1, SampleFormat::I8,
                &[ b'h', b'u', b'm', b'b', b'l', b'e' ], &[ 11, 12, 13, 14 ])?;
        assert_eq!(output, &[ b'.', b's', b'n', b'd',  0,0,0,32,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,
            b'h', b'u', b'm', b'b', b'l', b'e', 0, 0,
            11, 12, 13, 14 ]);

        // 3 desc padding bytes
        let output = write_desc(44100, 1, SampleFormat::I8,
                &[ b'W', b'o', b'r', b'l', b'd' ], &[ 11, 12, 13, 14 ])?;
        assert_eq!(output, &[ b'.', b's', b'n', b'd',  0,0,0,32,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,
            b'W', b'o', b'r', b'l', b'd', 0, 0, 0,
            11, 12, 13, 14 ]);
        Ok(())
    }

    #[test]
    fn test_write_samples_raw() -> AuResult<()> {
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I16,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.write_samples_raw(&[ 10, 128 ])?;
        assert!(writer.write_samples_i16(&[ 101, 102, 103 ]).is_err());
        writer.write_samples_raw(&[ 98, 12, 129, 99 ])?;
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,6,  0,0,0,3,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  10, 128, 98, 12, 129, 99 ]);
        Ok(())
    }

    #[test]
    fn test_flush() -> AuResult<()> {
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I8,
            channels: 1,
            ..AuWriteInfo::default()
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.flush()?;
        writer.write_samples_i8(&[ 10, 11 ])?;
        writer.flush()?;
        writer.write_samples_i8(&[ 12, 13 ])?;
        writer.flush()?;
        writer.finalize()?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_without_finalize() -> AuResult<()> {
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        assert_eq!(writer.get_mut().stream_position()?, 32);
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0xff,0xff,0xff,0xff,  0,0,0,2,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_finalize() -> AuResult<()> {
        // checks that data_size is updated to 4
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        writer.finalize()?;
        assert_eq!(writer.get_mut().stream_position()?, 32);
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,4,  0,0,0,2,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  10, 11, 12, 13 ]);

        // no samples written updates data_size to 0
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.finalize()?;
        assert_eq!(writer.get_mut().stream_position()?, 28);
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,0,  0,0,0,2,
            0,0,172,68,  0,0,0,1,  0,0,0,0 ]);
        Ok(())
    }

    #[test]
    fn test_finalize_update_position() -> AuResult<()> {
        // checks that the header is updated correctly relative to the start position
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I8,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        cursor.write_all(&[ 1, 2, 3, 4 ])?; // some garbage data
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.write_samples_i8(&[ 10, 11, 12, 13 ])?;
        writer.finalize()?;
        assert_eq!(writer.get_mut().stream_position()?, 36);
        assert_eq!(output, &[ 1, 2, 3, 4,
            b'.', b's', b'n', b'd',  0,0,0,28,  0,0,0,4,  0,0,0,2,  0,0,172,68,  0,0,0,1,  0,0,0,0,
            10, 11, 12, 13 ]);
        Ok(())
    }

    #[test]
    fn test_write_samples_i24() -> AuResult<()> {
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::I24,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        assert!(writer.write_samples_i8(&[ 0 ]).is_err());
        assert!(writer.write_samples_i16(&[ 0 ]).is_err());
        assert!(writer.write_samples_i32(&[ 0 ]).is_err());
        assert!(writer.write_samples_f32(&[ 0.0 ]).is_err());
        assert!(writer.write_samples_f64(&[ 0.0 ]).is_err());
        writer.write_samples_i24(&[ 1025, 65538 ])?;
        writer.write_samples_i24(&[ 13 ])?;
        assert!(writer.write_samples_raw(&[ 0 ]).is_err());
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0xff,0xff,0xff,0xff,  0,0,0,4,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  0, 4, 1,  1, 0, 2,  0, 0, 13 ]);
        Ok(())
    }

    #[test]
    fn test_write_samples_f32() -> AuResult<()> {
        let winfo = AuWriteInfo {
            sample_rate: 44100,
            sample_format: SampleFormat::F32,
            channels: 1,
        };
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        let mut writer = AuWriter::new(&mut cursor, &winfo)?;
        writer.write_samples_f32(&[ 1.0, -12.0 ])?;
        assert!(writer.write_samples_f64(&[ 0.0 ]).is_err());
        assert!(writer.write_samples_raw(&[ 0 ]).is_err());
        writer.write_samples_f32(&[ f32::NAN ])?;
        assert_eq!(output, [ b'.', b's', b'n', b'd',  0,0,0,28,  0xff,0xff,0xff,0xff,  0,0,0,6,
            0,0,172,68,  0,0,0,1,  0,0,0,0,  63, 128, 0, 0,  193, 64, 0, 0,  127, 192, 0, 0 ]);
        Ok(())
    }
}
