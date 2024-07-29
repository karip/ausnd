//! # AU Audio Format Reader and Writer
//!
//! A reader and writer for the
//! [AU audio format](https://en.wikipedia.org/wiki/Au_file_format).
//! AU is made of a small header, an optional description field and sample data.
//!
//! AU can store uncompressed integer or floating point sample data. It can also
//! store compressed sample data. This crate supports uncompressed sample data and
//! μ-law or A-law compressed sample data.
//!
//! The AU header contains the length of the sample data. The header can also
//! indicate that the length is unknown and that audio data should be read
//! until the end of the stream. This enables file lengths over 4 gigabytes and streams with
//! an infinite length.
//!
//! Provided functionality:
//!
//!  - [`AuReader`] reads the AU audio format, typically from a file.
//!  - [`AuWriter`] writes the AU audio format, typically to a file or a stream.
//!  - [`AuStreamParser`] parses the AU audio format and sends events to a callback.
//!    Useful for reading infinite AU streams.
//!
//! Terminology:
//! The term *info* is used to mean header data (made of channel count, sample rate, ...).
//! The term *description* is used to mean the description / info / annotation data field.

#![allow(
    clippy::question_mark,
)]
#![forbid(
    unsafe_code,
    clippy::panic,
    clippy::exit,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable,
)]
#![deny(
    clippy::cast_ptr_alignment,
    clippy::char_lit_as_u8,
    clippy::unnecessary_cast,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::checked_conversions,
)]

// silly way to test rust code blocks in README.md
// https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use std::io::{Read, Write, Seek, SeekFrom};

mod auresult;
mod aureader;
mod austreamparser;
mod auwriter;
mod cast;

pub use auresult::{AuResult, AuError};
pub use aureader::{AuReader, Sample, Samples};
pub use austreamparser::{AuStreamParser, AuEvent};
pub use auwriter::AuWriter;

/// Unknown data size in AU header.
const UNKNOWN_DATA_SIZE: u32 = 0xffffffff;

/// Sample format.
///
/// Unsupported sample formats are represented as Custom values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleFormat {
    /// Compressed μ-Law sample format. Each decoded sample is a signed 16-bit integer.
    /// Reading and writing samples should happen as signed 16-bit integers.
    /// The AU audio format encoding value 1.
    CompressedUlaw,
    /// Signed 8-bit integer sample format. The AU audio format encoding value 2.
    I8,
    /// Signed big-endian 16-bit integer sample format. The AU audio format encoding value 3.
    I16,
    /// Signed big-endian 24-bit integer sample format. The AU audio format encoding value 4.
    I24,
    /// Signed big-endian 32-bit integer sample format. The AU audio format encoding value 5.
    I32,
    /// Signed 32-bit floating point sample format. The AU audio format encoding value 6.
    F32,
    /// Signed 64-bit floating point sample format. The AU audio format encoding value 7.
    F64,
    /// Compressed A-Law sample format. Each decoded sample is a signed 16-bit integer.
    /// Reading and writing samples should happen as signed 16-bit integers.
    /// The AU audio format encoding value 27.
    CompressedAlaw,
    /// Custom: unsupported encoding. The inner u32 value is the AU audio format encoding value.
    /// The samples can be read and written only as raw data.
    Custom(u32),
}

impl SampleFormat {
    /// Returns the size of the decoded sample in bytes.
    /// Custom encodings always return 0.
    pub fn decoded_size(&self) -> usize {
        match &self {
            SampleFormat::I8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 => 4,
            SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
            SampleFormat::CompressedUlaw => 2,
            SampleFormat::CompressedAlaw => 2,
            SampleFormat::Custom(_) => 0,
        }
    }

    /// Returns the size of the encoded sample in the stream in bytes.
    #[inline(always)]
    fn bytesize_u8(&self) -> u8 {
        match &self {
            SampleFormat::I8 => 1,
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 => 4,
            SampleFormat::F32 => 4,
            SampleFormat::F64 => 8,
            SampleFormat::CompressedUlaw => 1,
            SampleFormat::CompressedAlaw => 1,
            SampleFormat::Custom(_) => 1,
        }
    }

    /// Returns the size of the encoded sample in the stream in bytes.
    #[inline(always)]
    fn bytesize_u64(&self) -> u64 {
        u64::from(self.bytesize_u8())
    }

    /// Returns the encoding value used by the AU format.
    fn as_u32(&self) -> u32 {
        match &self {
            SampleFormat::CompressedUlaw => 1,
            SampleFormat::I8 => 2,
            SampleFormat::I16 => 3,
            SampleFormat::I24 => 4,
            SampleFormat::I32 => 5,
            SampleFormat::F32 => 6,
            SampleFormat::F64 => 7,
            SampleFormat::CompressedAlaw => 27,
            SampleFormat::Custom(val) => *val,
        }
    }

    /// Converts the encoding value used by the AU format to `SampleFormat`.
    fn from_u32(value: u32) -> SampleFormat {
        match value {
            1 => SampleFormat::CompressedUlaw,
            2 => SampleFormat::I8,
            3 => SampleFormat::I16,
            4 => SampleFormat::I24,
            5 => SampleFormat::I32,
            6 => SampleFormat::F32,
            7 => SampleFormat::F64,
            27 => SampleFormat::CompressedAlaw,
            _ => SampleFormat::Custom(value),
        }
    }
}

// AU header size. The format docs says that header should be minimum 28 bytes, but many
// apps and libs write only 24 byte headers, so let's allow that.
const HEADER_SIZE: u8 = 24;

/// Audio info returned by `AuReader`.
#[derive(Debug, Clone, PartialEq)]
pub struct AuReadInfo {

    /// Number of channels.
    pub channels: u32,

    /// Sample rate, samples per second, e.g., 44100 or 48000.
    pub sample_rate: u32,

    /// Sample format.
    pub sample_format: SampleFormat,

    /// Length of the description in bytes.
    pub description_byte_len: u32,

    /// Sample count. The value is `None` for unsupported encodings or
    /// if the header indicated that the stream has an unknown length.
    pub sample_len: Option<u64>,

    /// Length of the sample data in bytes.
    /// The value is `None` if the header indicates that the stream has an unknown length.
    pub sample_byte_len: Option<u32>
}

impl AuReadInfo {
    /// Parses header and returns (offset, data_size, header).
    fn parse_bytes(header: &[u8; 24]) -> AuResult<(u64, AuReadInfo)> {
        if header[0] != b'.' || header[1] != b's' || header[2] != b'n' || header[3] != b'd' {
            return Err(AuError::UnrecognizedFormat);
        }
        let offset     = u32::from_be_bytes([ header[4], header[5], header[6], header[7] ]);
        let data_size  = u32::from_be_bytes([ header[8], header[9], header[10], header[11] ]);
        let encoding   = u32::from_be_bytes([ header[12], header[13], header[14], header[15] ]);
        let sample_rate= u32::from_be_bytes([ header[16], header[17], header[18], header[19] ]);
        let channels   = u32::from_be_bytes([ header[20], header[21], header[22], header[23] ]);
        if offset < 24 {
            return Err(AuError::InvalidAudioDataOffset);
        }
        let sample_format = SampleFormat::from_u32(encoding);
        let (sample_len, sample_byte_len) = if data_size != UNKNOWN_DATA_SIZE {
            let slen = if let SampleFormat::Custom(_) = sample_format {
                None
            } else {
                // division rounded down so that the possible last broken sample isn't included
                Some(u64::from(data_size) / sample_format.bytesize_u64())
            };
            (slen, Some(data_size))
        } else {
            (None, None)
        };
        Ok((u64::from(offset), AuReadInfo {
            channels,
            sample_rate,
            sample_format,
            description_byte_len: offset - u32::from(HEADER_SIZE),
            sample_byte_len,
            sample_len
        }))
    }
}

/// Audio info for `AuWriter`.
#[derive(Debug, Clone, PartialEq)]
pub struct AuWriteInfo {

    /// Number of channels.
    pub channels: u32,

    /// Sample rate, samples per second, e.g., 44100 or 48000.
    pub sample_rate: u32,

    /// Sample format.
    pub sample_format: SampleFormat,
}

impl Default for AuWriteInfo {
    /// Default values: 2 channels, sample rate 44100 and sample format I16.
    fn default() -> Self {
        AuWriteInfo {
            channels: 2,
            sample_rate: 44100,
            sample_format: SampleFormat::I16,
        }
    }
}

/// Checks if the given data is the start of an AU audio stream.
///
/// Only the first 24 bytes are checked. If the data length is less than 24 bytes,
/// then the result is always false.
///
/// # Examples
///
/// ```
/// if !ausnd::recognize(b"xsnd-some-invalid-au-data") {
///     println!("Not AU");
/// }
/// ```
pub fn recognize(data: &[u8]) -> bool {
    if data.len() < 24 ||
        data[0] != b'.' || data[1] != b's' || data[2] != b'n' || data[3] != b'd' {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recognize() {
        assert_eq!(recognize(&[]), false);
        assert_eq!(recognize(b".snd"), false);
        assert_eq!(recognize(b".snd4567890123456789012"), false);
        assert_eq!(recognize(b".snd45678901234567890123"), true);
        assert_eq!(recognize(b",snd45678901234567890123"), false);
    }
}
