
use crate::{cast, AuError, AuReadInfo, AuResult, Read, SampleFormat, Seek, SeekFrom};

/// Reader state.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ReadState {
    Initialized,
    InfoProcessed,
    DescriptionProcessed,
    ReadingSamples,
}

/// Sample data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sample {
    /// Signed 8-bit integer sample.
    I8(i8),
    /// Signed 16-bit integer sample.
    I16(i16),
    /// Signed 24-bit integer sample. The `i32` value is always in the range [-8388608, 8388607].
    I24(i32),
    /// Signed 32-bit integer sample.
    I32(i32),
    /// Signed 32-bit floating point sample.
    F32(f32),
    /// Signed 64-bit floating point sample.
    F64(f64)
}

/// The SampleRead trait allows for reading samples from a source.
trait SampleRead {
    /// Reads the next sample.
    fn read_sample_for_iter(&mut self) -> Option<AuResult<Sample>>;
    /// Returns a tuple where the first and second elements are the remaining sample count.
    /// For unknown sized audio data, returns (0, None).
    fn size_hint(&self) -> (usize, Option<usize>);
}

/// Iterator to read samples one by one.
#[derive(Debug)]
pub struct Samples<'a, R> {
    reader: &'a mut R
}

impl<'a, R> Samples<'a, R> {
    /// Creates a new sample iterator.
    fn new(reader: &'a mut R) -> Samples<R> {
        Samples {
            reader
        }
    }
}

/// Iterator implementation for samples.
impl<'a, R: SampleRead> Iterator for Samples<'a, R> {
    type Item = AuResult<Sample>;

    /// Reads the next sample.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.reader.read_sample_for_iter()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.reader.size_hint()
    }
}

/// AU audio format reader.
///
/// The methods should be called in this order:
///  - [`new()`](AuReader::new()) to create the reader
///  - [`read_info()`](AuReader::read_info()) to read the header
///  - optional: [`read_description()`](AuReader::read_description()) to read the description data
///  - [`read_sample()`](AuReader::read_sample()) or [`samples()`](AuReader::samples()) to
///    read sample data
///
/// When reading samples, channel data is interleaved.
/// The stream can also be seeked to a specific sample position.
///
/// The underlying reader must implement `Read + Seek`.
///
/// When a reader is created, it resolves unknown audio lengths from the stream length
/// so that the actual length is always known.
/// The reader can read and seek audio streams up to `u64::MAX` bytes (16_777_215 terabytes).
/// It is a playback duration of hundreds of years.
///
/// All data reading is done on demand. The reader doesn't perform any buffering, so
/// it's recommended to use a buffered reader with it.
///
/// # Errors
///
/// If any of the methods returns an error, then the reader is in an undefined state and
/// shouldn't be used anymore.
///
/// # Examples
///
/// Reading samples:
///
/// ```no_run
/// # fn example() -> ausnd::AuResult<()> {
/// let mut bufrd = std::io::BufReader::new(std::fs::File::open("test.au")?);
/// let mut reader = ausnd::AuReader::new(&mut bufrd)?;
/// let info = reader.read_info()?;
/// for s in reader.samples()? {
///     println!("Got sample {:?}", s?);
/// }
/// # Ok(())
/// # }
/// ```
///
/// Reading sample data as raw bytes (useful for reading unsupported encodings):
///
/// ```no_run
/// # use std::io::Read;
/// # fn example() -> ausnd::AuResult<()> {
/// let mut bufrd = std::io::BufReader::new(std::fs::File::open("test.au")?);
/// let mut reader = ausnd::AuReader::new(&mut bufrd)?;
/// let info = reader.read_info()?;
/// let size = usize::try_from(reader.resolved_sample_byte_len()?).expect("invalid size");
/// reader.seek_to_start_of_samples()?;
/// let mut ireader = reader.into_inner();
/// let mut raw_sample_data = vec![0u8; size];
/// ireader.read_exact(&mut raw_sample_data)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AuReader<R> {
    /// Read state.
    state: ReadState,

    /// Header once it has been read.
    info: Option<AuReadInfo>,

    /// Initial stream pos in bytes for read_info().
    initial_stream_pos: u64,

    /// Initial stream length in bytes for read_info().
    initial_stream_len: Option<u64>,

    /// Number of description bytes to be read.
    desc_bytes_unread_len: u64,

    /// Sample data start position in bytes relative to the header start. Value from the stream.
    sample_data_start_pos: u64,

    /// Sample read position in bytes relative to the header start.
    sample_data_read_pos: u64,

    /// Resolved sample data length in bytes.
    resolved_data_len: Option<u64>,

    /// The underlying reader.
    handle: R,
}

impl<R: Read> AuReader<R> {

    /* TODO: rename to new() and make public when Rust supports some kind of specialization. */
    /// Creates a new AuReader for inner reader implementing `Read`.
    #[allow(dead_code)]
    fn new_read(inner: R) -> AuResult<AuReader<R>> {
        // TODO: if inner implements Seek, find the end of the stream by seeking to its end
        //let initpos = inner.stream_position()?;
        //let initlen = inner.seek(SeekFrom::End(0))?;
        //inner.seek(SeekFrom::Start(initpos))?;
        let r = AuReader {
            state: ReadState::Initialized,
            info: None,
            initial_stream_pos: 0,
            initial_stream_len: None,
            desc_bytes_unread_len: 0,
            sample_data_start_pos: 0,
            sample_data_read_pos: 0,
            resolved_data_len: None,
            handle: inner,
        };
        Ok(r)
    }

    /// Reads [`AuReadInfo`] from the stream. The data is validated and an error is
    /// returned if the stream contains invalid data.
    pub fn read_info(&mut self) -> AuResult<AuReadInfo> {
        if self.info.is_some() {
            return Err(AuError::InvalidReadState);
        }
        let mut header = [0u8; crate::HEADER_SIZE as usize];
        self.handle.read_exact(&mut header)?;

        let (offset, h) = AuReadInfo::parse_bytes(&header)?;
        self.sample_data_start_pos = offset;
        self.sample_data_read_pos = offset;
        self.info = Some(h.clone());
        self.state = ReadState::InfoProcessed;
        // resolve the real data size from header or stream length
        if let Some(sbl) = h.sample_byte_len {
            self.resolved_data_len = Some(u64::from(sbl));

        } else if let Some(isl) = self.initial_stream_len {
            self.resolved_data_len = Some(isl - self.sample_data_start_pos -
                    self.initial_stream_pos);
        }
        self.desc_bytes_unread_len = self.sample_data_start_pos - u64::from(crate::HEADER_SIZE);
        Ok(h)
    }

    /// Pull some description bytes into the specified buffer, returning how many bytes were read.
    /// If the returned value is 0, then there are no more bytes to read or
    /// `out_buf` was 0 bytes in length. If the returned value is smaller than
    /// the length of the specified buffer, then this method should be called again
    /// until 0 is returned.
    ///
    /// This method can be called after [`read_info()`](AuReader::read_info()) and before
    /// samples have been read.
    ///
    /// The description bytes may contain binary data or ASCII characters with or
    /// without nul-termination.
    ///
    /// This method is implemented by reading bytes from the underlying reader and returns
    /// an error if that reading fails.
    pub fn read_description(&mut self, out_buf: &mut [u8]) -> AuResult<usize> {
        // if description or samples have already been read, don't read description
        if self.state != ReadState::InfoProcessed && self.state != ReadState::DescriptionProcessed {
            return Err(AuError::InvalidReadState);
        }
        if out_buf.is_empty() || self.desc_bytes_unread_len == 0 {
            return Ok(0);
        }
        let out_buf_len_u64 = u64::try_from(out_buf.len()).unwrap_or(u64::MAX);
        let read_buf = if out_buf_len_u64 <= self.desc_bytes_unread_len {
            out_buf
        } else {
            let desc_len_usize = usize::try_from(self.desc_bytes_unread_len).unwrap_or(usize::MAX);
            &mut out_buf[0..desc_len_usize]
        };
        let rs = self.handle.read(read_buf)?;
        self.desc_bytes_unread_len -= u64::try_from(rs).unwrap_or(self.desc_bytes_unread_len);
        if self.desc_bytes_unread_len == 0 {
            self.state = ReadState::DescriptionProcessed;
        }
        Ok(rs)
    }

    /// Seeks the stream to the start of the sample data. This method can only seek forward.
    /// Therefore, it returns an error if samples have already been read or
    /// [`seek()`](AuReader::seek()) has been called.
    ///
    /// This method has been implemented by reading bytes from the underlying reader
    /// and returns an error if it fails.
    pub fn seek_to_start_of_samples(&mut self) -> AuResult<()> {
        if self.state == ReadState::Initialized || self.state == ReadState::ReadingSamples {
            return Err(AuError::InvalidReadState);
        }
        self.skip_description()
    }

    /// Skips description bytes in the stream.
    fn skip_description(&mut self) -> AuResult<()> {
        if self.state != ReadState::InfoProcessed {
            return Ok(());
        }
        let mut skip_buf = [0u8; 1];
        for _ in 0..self.desc_bytes_unread_len {
            self.handle.read_exact(&mut skip_buf)?;
        }
        self.desc_bytes_unread_len = 0;
        self.state = ReadState::DescriptionProcessed;
        Ok(())
    }

    /// Reads one sample.
    ///
    /// Returns `None` when all samples have been read.
    /// Returns an error for unsupported encodings (`SampleFormat::Custom`) or if reading from
    /// the underlying reader fails.
    #[inline(always)]
    pub fn read_sample(&mut self) -> AuResult<Option<Sample>> {
        if self.state != ReadState::ReadingSamples {
            if self.state == ReadState::Initialized {
                return Err(AuError::InvalidReadState);
            }
            let Some(info) = &self.info else {
                return Err(AuError::InvalidReadState);
            };
            if let SampleFormat::Custom(_) = info.sample_format {
                return Err(AuError::Unsupported);
            }
            match self.skip_description() {
                Ok(()) => {},
                Err(e) => { return Err(e); }
            }
            self.state = ReadState::ReadingSamples;
        }
        let Some(info) = &self.info else {
            return Err(AuError::InvalidReadState);
        };
        let sf = info.sample_format;
        let bsize = sf.bytesize_u64();
        // if the header has known size, read until that size and then stop
        if let Some(slen) = self.resolved_data_len {
            if self.sample_data_read_pos - self.sample_data_start_pos + bsize > slen {
                return Ok(None);
            }
        }
        let sample = match self.read_sample_for_format(sf) {
            Ok(Some(s)) => Ok(Some(s)),
            Ok(None) => {
                if self.resolved_data_len.is_none() {
                    // if the stream is appended, then reading might continue which is not great
                    // because a partial sample may have already been read and the result would
                    // be bad samples. so, set the current read pos as the data len.
                    self.resolved_data_len = Some(self.sample_data_read_pos -
                            self.sample_data_start_pos);
                    Ok(None)
                } else {
                    // return error here so that sample_data_read_pos isn't incremented
                    return Err(AuError::StdIoError(
                        std::io::Error::from(std::io::ErrorKind::UnexpectedEof)));
                }
            },
            Err(e) => {
                // return error here so that sample_data_read_pos isn't incremented
                return Err(e);
            },
        };
        self.sample_data_read_pos = self.sample_data_read_pos.saturating_add(bsize);
        sample
    }

    fn read_sample_for_format(&mut self, sample_format: SampleFormat) -> AuResult<Option<Sample>> {
        match sample_format {
            SampleFormat::I8 => self.read_sample_i8(),
            SampleFormat::I16 => self.read_sample_i16(),
            SampleFormat::I24 => self.read_sample_i24(),
            SampleFormat::I32 => self.read_sample_i32(),
            SampleFormat::F32 => self.read_sample_f32(),
            SampleFormat::F64 => self.read_sample_f64(),
            SampleFormat::CompressedUlaw => {
                let val = self.read_sample_u8()?;
                Ok(Some(Sample::I16(audio_codec_algorithms::decode_ulaw(val))))
            },
            SampleFormat::CompressedAlaw => {
                let val = self.read_sample_u8()?;
                Ok(Some(Sample::I16(audio_codec_algorithms::decode_alaw(val))))
            },
            SampleFormat::Custom(_) => {
                Err(AuError::Unsupported)
            }
        }
    }

    fn read_sample_u8(&mut self) -> AuResult<u8> {
        let mut buf = [ 0u8; 1 ];
        self.handle.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_sample_i8(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 1 ];
        self.handle.read_exact(&mut buf)?;
        Ok(Some(Sample::I8(cast::u8_as_i8(buf[0]))))
    }

    fn read_sample_i16(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 2 ];
        // if reading the first byte returns eof, then this is the end of the stream
        match self.handle.read_exact(&mut buf[0..1]) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => {
                return Err(e)?;
            },
        }
        // if other bytes return eof, it's a broken sample
        self.handle.read_exact(&mut buf[1..2])?;
        Ok(Some(Sample::I16(i16::from_be_bytes(buf))))
    }

    fn read_sample_i24(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 3 ];
        // if reading the first byte returns eof, then this is the end of the stream
        match self.handle.read_exact(&mut buf[0..1]) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => {
                return Err(e)?;
            },
        }
        // if other bytes return eof, it's a broken sample
        self.handle.read_exact(&mut buf[1..3])?;
        let mut res = i32::from(buf[0]) << 16 | i32::from(buf[1]) << 8 | i32::from(buf[2]);
        if res >= 8388608 {
            res -= 16777216;
        }
        Ok(Some(Sample::I24(res)))
    }

    fn read_sample_i32(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 4 ];
        // if reading the first byte returns eof, then this is the end of the stream
        match self.handle.read_exact(&mut buf[0..1]) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => {
                return Err(e)?;
            },
        }
        // if other bytes return eof, it's a broken sample
        self.handle.read_exact(&mut buf[1..4])?;
        Ok(Some(Sample::I32(i32::from_be_bytes(buf))))
    }

    fn read_sample_f32(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 4 ];
        // if reading the first byte returns eof, then this is the end of the stream
        match self.handle.read_exact(&mut buf[0..1]) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => {
                return Err(e)?;
            },
        }
        // if other bytes return eof, it's a broken sample
        self.handle.read_exact(&mut buf[1..4])?;
        Ok(Some(Sample::F32(f32::from_be_bytes(buf))))
    }

    fn read_sample_f64(&mut self) -> AuResult<Option<Sample>> {
        let mut buf = [ 0u8; 8 ];
        // if reading the first byte returns eof, then this is the end of the stream
        match self.handle.read_exact(&mut buf[0..1]) {
            Ok(()) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            },
            Err(e) => {
                return Err(e)?;
            },
        }
        // if other bytes return eof, it's a broken sample
        self.handle.read_exact(&mut buf[1..8])?;
        Ok(Some(Sample::F64(f64::from_be_bytes(buf))))
    }

    /// Returns an iterator for samples. Returns an error for unsupported encodings
    /// (`SampleFormat::Custom`).
    ///
    /// Note: calling `next()` for [`Samples`] may return an error, which should be checked.
    /// Otherwise, the iterator may return an infinite number of errors and loop forever:
    ///
    /// ```ignore
    /// reader.samples().count() // don't do this, loops forever if an error is encountered
    /// reader.samples().map(|s| s.expect("error")).count() // better, panics for errors
    /// ```
    pub fn samples(&mut self) -> AuResult<Samples<'_, AuReader<R>>> {
        if self.state == ReadState::Initialized {
            return Err(AuError::InvalidReadState);
        }
        let Some(info) = &self.info else {
            return Err(AuError::InvalidReadState);
        };
        if let SampleFormat::Custom(_) = info.sample_format {
            return Err(AuError::Unsupported);
        }
        self.skip_description()?;
        self.state = ReadState::ReadingSamples;

        Ok(Samples::new(self))
    }

    /// Consumes this `AuReader` and returns the underlying reader.
    pub fn into_inner(self) -> R {
        self.handle
    }

    /// Gets a reference to the underlying reader.
    ///
    /// It is not recommended to directly read from the underlying reader.
    pub const fn get_ref(&self) -> &R {
        &self.handle
    }

    /// Gets a mutable reference to the underlying reader.
    ///
    /// Care should be taken to avoid modifying the internal I/O state of the
    /// underlying reader as it may corrupt this reader's state.
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.handle
    }
}

impl<R: Read> SampleRead for AuReader<R> {
    #[inline(always)]
    fn read_sample_for_iter(&mut self) -> Option<AuResult<Sample>> {
        match self.read_sample() {
            Ok(Some(s)) => Some(Ok(s)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // get the lower and upper bound of the sample count
        let Some(info) = &self.info else {
            return (0, None);
        };
        let sample_size = info.sample_format.bytesize_u64();
        let Some(slen) = self.resolved_data_len else { // unknown size
            return (0, None);
        };
        let sample_count = (self.sample_data_start_pos + slen - self.sample_data_read_pos) /
            sample_size;
        if let Ok(ss) = usize::try_from(sample_count) {
            (ss, Some(ss))
        } else { // sample count is greater than usize::MAX
            (usize::MAX, None)
        }
    }
}

impl<R: Read + Seek> AuReader<R> {
    /// Creates a new AuReader for the inner reader implementing `Read + Seek`.
    pub fn new(mut inner: R) -> AuResult<AuReader<R>> {
        // find the end of the stream by seeking to its end
        let initpos = inner.stream_position()?;
        let initlen = inner.seek(SeekFrom::End(0))?;
        inner.seek(SeekFrom::Start(initpos))?;
        let r = AuReader {
            state: ReadState::Initialized,
            info: None,
            initial_stream_pos: initpos,
            initial_stream_len: Some(initlen),
            desc_bytes_unread_len: 0,
            sample_data_start_pos: 0,
            sample_data_read_pos: 0,
            resolved_data_len: None,
            handle: inner,
        };
        Ok(r)
    }

    /// Returns the total number of samples in the stream.
    /// Returns an error for custom sample formats or if [`read_info()`](AuReader::read_info())
    /// hasn't been called.
    ///
    /// The difference between this method and [`AuReadInfo::sample_len`] is that this
    /// method always returns the length even if the length is unknown in the header.
    pub fn resolved_sample_len(&self) -> AuResult<u64> {
        let Some(info) = &self.info else {
            return Err(AuError::InvalidReadState);
        };
        let Some(slen) = self.resolved_data_len else {
            return Err(AuError::InvalidReadState);
        };
        if let SampleFormat::Custom(_) = info.sample_format {
            return Err(AuError::Unsupported);
        }
        // division rounded down so that the possible last broken sample isn't included
        Ok(slen / info.sample_format.bytesize_u64())
    }

    /// Returns the total number of sample bytes in the stream.
    /// Returns an error if [`read_info()`](AuReader::read_info()) hasn't been called.
    ///
    /// The difference between this method and [`AuReadInfo::sample_byte_len`] is that this
    /// method always returns the length even if the length is unknown in the header.
    pub fn resolved_sample_byte_len(&self) -> AuResult<u64> {
        let Some(slen) = self.resolved_data_len else {
            return Err(AuError::InvalidReadState);
        };
        Ok(slen)
    }

    /// Seeks the stream to the given sample position.
    /// Returns an error if trying to seek past the maximum sample position.
    pub fn seek(&mut self, sample_position: u64) -> AuResult<()> {
        let Some(info) = &self.info else {
            return Err(AuError::InvalidReadState);
        };
        let sample_size = info.sample_format.bytesize_u64();
        if let SampleFormat::Custom(_) = info.sample_format {
            return Err(AuError::Unsupported);
        }
        if self.state == ReadState::InfoProcessed {
            self.skip_description()?;
        }
        let total_samples = match self.resolved_sample_len() {
            Ok(sl) => sl,
            Err(_) => { return Err(AuError::SeekError); }
        };
        if sample_position > total_samples {
            return Err(AuError::SeekError);
        }
        let Some(new_pos) = sample_position.checked_mul(sample_size)
            .and_then(|val| val.checked_add(self.sample_data_start_pos)) else {
            return Err(AuError::SeekError);
        };
        let Some(seek_offset) = i64::try_from(new_pos)
            .ok()
            .and_then(|p| p.checked_sub_unsigned(self.sample_data_read_pos)) else {
            return Err(AuError::SeekError);
        };
        self.handle.seek(SeekFrom::Current(seek_offset))?;
        self.sample_data_read_pos = new_pos;
        self.state = ReadState::ReadingSamples;
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    /// Reader, which can be appended.
    struct AppendableReader {
        data: Vec<u8>,
        pos: u64,
    }

    impl AppendableReader {
        pub fn new() -> AppendableReader {
            AppendableReader {
                data: vec![],
                pos: 0
            }
        }

        pub fn append(&mut self, data: &[u8]) {
            self.data.extend_from_slice(data);
        }

        pub fn clear(&mut self) {
            self.data.clear();
        }
    }

    impl Read for AppendableReader {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize, std::io::Error> {
            let Ok(mut p) = usize::try_from(self.pos) else {
                return Err(std::io::Error::from(std::io::ErrorKind::Other));
            };
            if self.data.len() < p {
                return Err(std::io::Error::from(std::io::ErrorKind::Other));
            }
            let len = buf.len().min(self.data.len() - p);
            for b in buf.iter_mut().take(len) {
                *b = self.data[p];
                p += 1;
                self.pos += 1;
            }
            Ok(len)
        }
    }

    impl Seek for AppendableReader {
        fn seek(&mut self, sf: SeekFrom) -> std::io::Result<u64> {
            match sf {
                SeekFrom::Start(p) => { self.pos = p; },
                SeekFrom::End(p) => {
                    let Ok(dlen) = i64::try_from(self.data.len()) else {
                        return Err(std::io::Error::from(std::io::ErrorKind::Other));
                    };
                    self.pos = u64::try_from(dlen + p)
                        .map_err(|_| std::io::Error::from(std::io::ErrorKind::Other))?;
                },
                SeekFrom::Current(p) => {
                    self.pos = match self.pos.checked_add_signed(p) {
                        Some(v) => v,
                        None => { return Err(std::io::Error::from(std::io::ErrorKind::Other)); }
                    }
                },
            }
            Ok(self.pos)
        }
    }

    #[test]
    fn test_appendable_reader() -> AuResult<()> {
        let mut ar = AppendableReader::new();
        ar.append(&[ 1, 2, 3, 4, 5, 6 ]);
        let mut buf = [0u8; 20];
        assert_eq!(ar.read(&mut buf)?, 6);
        assert_eq!(buf[0..6], [ 1, 2, 3, 4, 5, 6 ]);
        assert_eq!(ar.read(&mut buf)?, 0);
        ar.append(&[ 7, 8, 9 ]);
        assert_eq!(ar.read(&mut buf)?, 3);
        assert_eq!(buf[0..3], [ 7, 8, 9 ]);
        assert_eq!(ar.read(&mut buf)?, 0);
        ar.append(&[ 10, 11 ]);
        assert_eq!(ar.read(&mut buf)?, 2);
        assert_eq!(buf[0..2], [ 10, 11 ]);
        assert_eq!(ar.read(&mut buf)?, 0);
        assert_eq!(ar.read(&mut buf)?, 0);
        ar.append(&[ 12 ]);
        assert_eq!(ar.read(&mut buf)?, 1);
        assert_eq!(buf[0..1], [ 12 ]);
        assert_eq!(ar.read(&mut buf)?, 0);
        Ok(())
    }

    fn create_au_hdr(dummy_data_len: usize, data_size: u32, format: SampleFormat,
            rate: u32, channels: u32) -> Vec<u8> {
        create_au_hdr_with_desc(dummy_data_len, data_size, format, rate, channels, &[])
    }

    fn create_au_hdr_with_desc(dummy_data_len: usize, data_size: u32, format: SampleFormat,
            rate: u32, channels: u32, desc: &[u8]) -> Vec<u8> {
        let header_size = (24 + desc.len()) as u32;
        let mut data = vec![];
        data.extend_from_slice(&[ 0xff, 0xfe, 0xfd, 0xfc ][0..dummy_data_len]);
        data.extend_from_slice(&[ b'.', b's', b'n', b'd' ]);    // magic
        data.extend_from_slice(&header_size.to_be_bytes());     // offset
        data.extend_from_slice(&data_size.to_be_bytes());       // data size
        data.extend_from_slice(&format.as_u32().to_be_bytes()); // encoding
        data.extend_from_slice(&rate.to_be_bytes());            // sample rate
        data.extend_from_slice(&channels.to_be_bytes());        // channels
        data.extend_from_slice(&desc);                          // description
        data
    }

    fn read_dummy_data(cursor: &mut impl Read, len: usize) -> AuResult<()> {
        let mut init_buf = vec![0u8; len];
        Ok(cursor.read_exact(&mut init_buf)?)
    }

    #[test]
    fn test_new_transferring_ownership() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        Ok(())
    }

    #[test]
    fn test_new_taking_mut_ref() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let mut cursor = Cursor::new(&au);
        let mut reader = AuReader::new(&mut cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        Ok(())
    }

    #[test]
    fn test_new_ref_slice() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let mut rd: &[u8] = au.as_ref();
        let mut reader = AuReader::new_read(&mut rd)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        Ok(())
    }

    #[test]
    fn test_into_inner() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.samples()?.next().expect("sample read error")?, Sample::I8(11));
        let mut cr = reader.into_inner();
        let mut buf = [0u8; 1];
        assert_eq!(cr.read(&mut buf)?, 1);
        assert_eq!(buf[0], 12);
        Ok(())
    }

    #[test]
    fn test_get_ref() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        let cr = reader.get_ref();
        assert_eq!(cr.position(), 24);
        assert_eq!(reader.samples()?.next().expect("sample read error")?, Sample::I8(11));
        Ok(())
    }

    #[test]
    fn test_get_mut() -> AuResult<()> {
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::I8);
        assert_eq!(reader.samples()?.next().expect("sample read error")?, Sample::I8(11));
        let cr = reader.get_mut();
        let mut buf = [0u8; 1];
        assert_eq!(cr.read(&mut buf)?, 1);
        assert_eq!(buf[0], 12);
        Ok(())
    }

    #[test]
    fn test_read_info() -> AuResult<()> {
        for data_size in [16, 0xffffffff] {
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I16, 44100, 1, &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.channels, 1);
            assert_eq!(info.sample_format, SampleFormat::I16);
            assert_eq!(info.sample_rate, 44100);
            assert_eq!(info.description_byte_len, 4);
            if data_size != 0xffffffff {
                assert_eq!(info.sample_len, Some(8));
                assert_eq!(info.sample_byte_len, Some(16));
            } else {
                assert_eq!(info.sample_len, None);
                assert_eq!(info.sample_byte_len, None);
            }
        }
        Ok(())
    }

    #[test]
    fn test_read_info_for_unsupported_encoding() -> AuResult<()> {
        for data_size in [16, 0xffffffff] {
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::Custom(0x402), 44100, 1, &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.channels, 1);
            assert_eq!(info.sample_format, SampleFormat::Custom(0x402));
            assert_eq!(info.sample_rate, 44100);
            assert_eq!(info.description_byte_len, 4);
            // sample len is always None
            assert_eq!(info.sample_len, None);
            if data_size != 0xffffffff {
                assert_eq!(info.sample_byte_len, Some(16));
            } else {
                assert_eq!(info.sample_byte_len, None);
            }
        }
        Ok(())
    }

    #[test]
    fn test_read_description() -> AuResult<()> {
        for data_size in [8, 0xffffffff] {
            // desc empty
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            assert!(reader.read_description(&mut []).is_err());
            let info = reader.read_info()?;
            assert_eq!(info.description_byte_len, 0);
            // description can be called many times for an empty buf
            assert_eq!(reader.read_description(&mut [])?, 0);
            assert_eq!(reader.read_description(&mut [])?, 0);
            // no bytes are read to the buffer
            let mut desc = [ 99, 99 ];
            assert_eq!(reader.read_description(&mut desc)?, 0);
            assert_eq!(desc, [ 99, 99 ]);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            // description can't be read after samples have been read
            assert!(reader.read_description(&mut []).is_err());

            // desc "WORLD"
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[ b'W', b'O', b'R', b'L', b'D', 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.channels, 1);
            assert_eq!(info.sample_format, SampleFormat::I8);
            assert_eq!(info.sample_rate, 44100);
            // description can be called many times for an empty buf
            assert_eq!(reader.read_description(&mut [])?, 0);
            assert_eq!(reader.read_description(&mut [])?, 0);
            let mut desc = vec![0u8; 4];
            assert_eq!(reader.read_description(&mut desc)?, 4);
            assert_eq!(desc, &[ b'W', b'O', b'R', b'L' ]);
            assert_eq!(reader.read_description(&mut desc)?, 2);
            assert_eq!(desc, &[ b'D', 0, b'R', b'L' ]);
            assert_eq!(reader.read_description(&mut desc)?, 0);
            assert_eq!(reader.read_description(&mut [])?, 0);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            // description can't be read to an empty buf after samples have been read
            assert!(reader.read_description(&mut []).is_err());
            // description can't be read after samples have been read
            assert!(reader.read_description(&mut desc).is_err());

            // desc four zeros
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            assert_eq!(reader.read_description(&mut [])?, 0);
            let mut desc = vec![0u8; 4];
            assert_eq!(reader.read_description(&mut desc)?, 4);
            assert_eq!(desc, &[ 0, 0, 0, 0 ]);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));

            // desc single byte: 99
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[ 99 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            let mut desc = vec![0u8; 4];
            assert_eq!(reader.read_description(&mut desc)?, 1);
            assert_eq!(desc, &[ 99, 0, 0, 0 ]);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));

            // desc 8 bytes
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[ 65, 66, 67, 68, 69, 70, 71, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            let mut desc = vec![0u8; 8];
            let _ = reader.read_description(&mut desc)?;
            assert_eq!(desc, &[ 65, 66, 67, 68, 69, 70, 71, 0 ]);
        }
        Ok(())
    }

    #[test]
    fn test_seek_to_start_of_samples() -> AuResult<()> {
        // failing call
        let mut au = create_au_hdr(0, 8, SampleFormat::Custom(0x402), 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        assert!(reader.seek_to_start_of_samples().is_err());

        // failing call after reading a sample
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.read_sample().expect("sample error"), Some(Sample::I8(11)));
        assert!(reader.seek_to_start_of_samples().is_err());

        // failing call after seek
        let mut au = create_au_hdr(0, 8, SampleFormat::I8, 44100, 1);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        let cursor = Cursor::new(&au);
        let mut reader = AuReader::new(cursor)?;
        let _ = reader.read_info()?;
        reader.seek(0)?;
        assert!(reader.seek_to_start_of_samples().is_err());

        // successful call
        let mut au = create_au_hdr_with_desc(3, 8, SampleFormat::Custom(0x402), 44100, 1, &[ 65, 65, 65, 0 ]);
        au.extend_from_slice(&[ 11, 12, 13, 14, 15, 16, 17, 18, 98, 99 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(cursor)?;
        let info = reader.read_info()?;
        assert_eq!(info.sample_format, SampleFormat::Custom(0x402));
        assert_eq!(info.sample_len, None);
        assert_eq!(reader.resolved_sample_byte_len().expect("len failed"), 8);
        assert!(reader.seek_to_start_of_samples().is_ok());
        let mut cr = reader.into_inner();
        let mut buf = [0u8; 8];
        assert_eq!(cr.read(&mut buf)?, 8);
        assert_eq!(buf, [ 11, 12, 13, 14, 15, 16, 17, 18 ]);
        Ok(())
    }

    #[test]
    fn test_read_samples_skips_description() -> AuResult<()> {
        for data_size in [32, 0xffffffff] {
            // description is skipped when reading samples
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                    &[ b'H', b'E', b'L', b'L', b'O', 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.channels, 1);
            assert_eq!(info.sample_format, SampleFormat::I8);
            assert_eq!(info.sample_rate, 44100);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            let mut desc = vec![0u8; 10];
            assert!(reader.read_description(&mut desc).is_err());

            // partially read description is skipped when reading samples
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1,
                &[ b'H', b'E', b'L', b'L', b'O', 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.channels, 1);
            assert_eq!(info.sample_format, SampleFormat::I8);
            assert_eq!(info.sample_rate, 44100);
            let mut desc = vec![0u8; 3];
            assert_eq!(reader.read_description(&mut desc)?, 3);
            assert_eq!(desc, &[ b'H', b'E', b'L' ]);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            let mut desc = vec![0u8; 10];
            assert!(reader.read_description(&mut desc).is_err());
        }
        Ok(())
    }

    #[test]
    fn test_read_sample() -> AuResult<()> {
        for data_size in [32, 0xffffffff] {
            let mut au = create_au_hdr(3, data_size, SampleFormat::I32, 44100, 1);
            au.extend_from_slice(&[ 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4,
                                    0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let info = reader.read_info()?;
            assert_eq!(info.sample_format, SampleFormat::I32);
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(1)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(2)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(3)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(4)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(5)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(6)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(7)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I32(8)));
            assert!(reader.read_sample().expect("sample read error").is_none());
        }
        Ok(())
    }

    #[test]
    fn test_read_samples_appending_new_data() -> AuResult<()> {
        let mut buffer = AppendableReader::new();
        let au = create_au_hdr(3, 0xffffffff, SampleFormat::I32, 44100, 1);
        buffer.append(&au);
        buffer.append(&[ 0, 0, 0, 1 ]);
        let mut init_buf = vec![0u8; 3];
        buffer.read_exact(&mut init_buf)?;
        let mut reader = AuReader::new(&mut buffer)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.samples()?.next().expect("sample read error")?, Sample::I32(1));
        // data appended won't make it readable
        reader.get_mut().append(&[ 0, 0, 0, 2 ]);
        assert!(reader.samples()?.next().is_none());

        let mut buffer = AppendableReader::new();
        let au = create_au_hdr(3, 0xffffffff, SampleFormat::I32, 44100, 1);
        buffer.append(&au);
        // one sample and a half broken sample
        buffer.append(&[ 0, 0, 0, 1, 0, 0 ]);
        let mut init_buf = vec![0u8; 3];
        buffer.read_exact(&mut init_buf)?;
        let mut reader = AuReader::new(&mut buffer)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.samples()?.next().expect("sample read error")?, Sample::I32(1));
        // data appended won't make it readable
        reader.get_mut().append(&[ 0, 2, 0, 0, 0, 3 ]);
        assert!(reader.samples()?.next().is_none());

        Ok(())
    }

    #[test]
    fn test_read_sample_for_broken_last_sample_read_only() -> AuResult<()> {
        // non-broken sample - no error
        let mut au = create_au_hdr_with_desc(3, 0xffffffff, SampleFormat::F32, 44100, 1, &[ 0, 0, 0, 0 ]);
        au.extend_from_slice(&[ 0, 0, 0, 0, ]);
        let mut rd: &[u8] = au.as_ref();
        read_dummy_data(&mut rd, 3)?;
        let mut reader = AuReader::new_read(&mut rd)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.read_sample()?, Some(Sample::F32(0.0)));
        assert_eq!(reader.read_sample()?, None);

        // the last sample is broken (not enough data)
        struct TestCase { sf: SampleFormat, sample: Sample, count: u32 }
        let test_cases: [TestCase; 5] = [
            TestCase { sf: SampleFormat::I16, sample: Sample::I16(0), count: 5 },
            TestCase { sf: SampleFormat::I24, sample: Sample::I24(0), count: 3 },
            TestCase { sf: SampleFormat::I32, sample: Sample::I32(0), count: 2 },
            TestCase { sf: SampleFormat::F32, sample: Sample::F32(0.0), count: 2 },
            TestCase { sf: SampleFormat::F64, sample: Sample::F64(0.0), count: 1 },
        ];
        for test_case in test_cases {
            let mut au = create_au_hdr_with_desc(3, 0xffffffff,
                test_case.sf, 44100, 1, &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]);
            let mut rd: &[u8] = au.as_ref();
            read_dummy_data(&mut rd, 3)?;
            let mut reader = AuReader::new_read(&mut rd)?;
            let _ = reader.read_info()?;
            for _ in 0..test_case.count {
                assert_eq!(reader.read_sample()?, Some(test_case.sample));
            }
            // the last sample is broken for every test_case,
            // so this call to read_sample() returns an error
            assert!(reader.read_sample().is_err());
        }
        Ok(())
    }

    #[test]
    fn test_read_samples_stream_cleared() -> AuResult<()> {
        // tests cases where the stream is suddenly cleared, creating errors
        let mut buffer = AppendableReader::new();
        let au = create_au_hdr(3, 0xffffffff, SampleFormat::I32, 44100, 1);
        buffer.append(&au);
        buffer.append(&[ 0, 0, 0, 1, 0, 0, 0, 2 ]);
        let mut init_buf = vec![0u8; 3];
        buffer.read_exact(&mut init_buf)?;
        let mut reader = AuReader::new(&mut buffer)?;
        // stream is cleared
        reader.get_mut().clear();
        assert!(reader.read_info().is_err());

        let mut buffer = AppendableReader::new();
        let au = create_au_hdr(3, 0xffffffff, SampleFormat::I32, 44100, 1);
        buffer.append(&au);
        buffer.append(&[ 0, 0, 0, 1, 0, 0, 0, 2 ]);
        let mut init_buf = vec![0u8; 3];
        buffer.read_exact(&mut init_buf)?;
        let mut reader = AuReader::new(&mut buffer)?;
        let _ = reader.read_info()?;
        // stream is cleared
        reader.get_mut().clear();
        assert!(reader.read_sample().is_err());
        assert!(reader.samples()?.next().expect("sample read error").is_err());
        // seek succeeds because std::io::Seek::seek() can seek to position after the end
        assert!(reader.seek(1).is_ok());
        assert!(reader.read_sample().is_err());

        Ok(())
    }

    #[test]
    fn test_samples_iterator_and_size_hint() -> AuResult<()> {
        for data_size in [8, 0xffffffff] {
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1, &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            // check that size_hint() is decreasing
            {
                let mut siter = reader.samples()?;
                assert_eq!(siter.size_hint(), (8, Some(8)));
                assert_eq!(siter.next().expect("sample read error")?, Sample::I8(1));
                assert_eq!(siter.size_hint(), (7, Some(7)));
            }
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(2)));
            {
                let siter = reader.samples()?;
                assert_eq!(siter.size_hint(), (6, Some(6)));
            }
            // ensure the remaining samples are read
            let expected = vec![
                Sample::I8(3), Sample::I8(4),
                Sample::I8(5), Sample::I8(6), Sample::I8(7), Sample::I8(8),
            ];
            let mut count = 0;
            for (i, sample) in reader.samples()?.enumerate() {
                assert_eq!(sample.expect("no sample"), expected[i]);
                count += 1;
            }
            assert_eq!(count, 6);
            // size_hint() and count() must be zero after all samples have been read
            {
                let siter = reader.samples()?;
                assert_eq!(siter.size_hint(), (0, Some(0)));
                assert_eq!(siter.count(), 0);
            }
            // reader can be accessed after the iterator isn't used anymore
            assert!(reader.read_sample().expect("read sample error").is_none());
            assert_eq!(reader.resolved_sample_len().expect("invalid len"), 8);
        }
        Ok(())
    }

    #[test]
    fn test_samples_iterator_and_size_hint_for_read_only() -> AuResult<()> {
        for data_size in [8, 0xffffffff] {
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1, &[ 0, 0, 0, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            let mut rd: &[u8] = au.as_ref();
            read_dummy_data(&mut rd, 3)?;
            let mut reader = AuReader::new_read(&mut rd)?;
            let _ = reader.read_info()?;
            // check that size_hint() is decreasing
            {
                let mut siter = reader.samples()?;
                if data_size != 0xffffffff {
                    assert_eq!(siter.size_hint(), (8, Some(8)));
                } else {
                    assert_eq!(siter.size_hint(), (0, None));
                }
                assert_eq!(siter.next().expect("sample read error")?, Sample::I8(1));
                if data_size != 0xffffffff {
                    assert_eq!(siter.size_hint(), (7, Some(7)));
                } else {
                    assert_eq!(siter.size_hint(), (0, None));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_resolved_sample_len_and_byte_len() -> AuResult<()> {
        let mut au = create_au_hdr(3, 8, SampleFormat::I16, 44100, 1);
        // audio data contains extra bytes, which should not be part of sample len
        au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.resolved_sample_len().expect("invalid len"), 4);
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 8);

        // test audio data size which isn't a multiple of the sample size
        let mut au = create_au_hdr(3, 9, SampleFormat::I16, 44100, 1);
        au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.resolved_sample_len().expect("invalid len"), 4);
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 9);

        // len is known even if the header has unknown size
        let mut au = create_au_hdr(3, 0xffffffff, SampleFormat::I16, 44100, 1);
        au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.resolved_sample_len().expect("invalid len"), 8);
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 16);

        // len is known even if the header has unknown size (not a multiple of the sample size)
        let mut au = create_au_hdr(3, 0xffffffff, SampleFormat::I16, 44100, 1);
        au.extend_from_slice(&[ 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert_eq!(reader.resolved_sample_len().expect("invalid len"), 7);
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 15);

        Ok(())
    }

    #[test]
    fn test_resolved_sample_len_and_byte_len_for_unsupported_encoding() -> AuResult<()> {
        let mut au = create_au_hdr(3, 9, SampleFormat::Custom(0x402), 44100, 1);
        // audio data contains extra bytes, which should not be part of sample len
        au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert!(reader.resolved_sample_len().is_err());
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 9);

        // len is known even if the header has unknown size
        let mut au = create_au_hdr(3, 0xffffffff, SampleFormat::Custom(0x402), 44100, 1);
        au.extend_from_slice(&[ 1, 2, 3, 4, 5 ]);
        let mut cursor = Cursor::new(&au);
        read_dummy_data(&mut cursor, 3)?;
        let mut reader = AuReader::new(&mut cursor)?;
        let _ = reader.read_info()?;
        assert!(reader.resolved_sample_len().is_err());
        assert_eq!(reader.resolved_sample_byte_len().expect("invalid len"), 5);

        Ok(())
    }

    #[test]
    fn test_seek() -> AuResult<()> {
        for data_size in [8, 0xffffffff] {
            // three dummy bytes at the start of the data to test that they don't affect seeking
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1, &[ 65, 65, 65, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            if data_size != 0xffffffff {
                // audio data contains extra bytes, which should not be part of sample data
                au.extend_from_slice(&[ 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);
            }
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(2)));
            reader.seek(0)?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(1)));
            reader.seek(4)?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(5)));
            reader.seek(7)?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(8)));
            reader.seek(8)?;
            assert!(reader.read_sample().expect("sample read error").is_none());
            assert!(reader.seek(9).is_err());
            assert!(reader.read_sample().expect("sample read error").is_none());
            assert!(reader.seek(u64::MAX).is_err());
            assert!(reader.read_sample().expect("sample read error").is_none());
            reader.seek(4)?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(5)));

            // check seek can be called before reading any samples
            let mut au = create_au_hdr_with_desc(3, data_size, SampleFormat::I8, 44100, 1, &[ 65, 65, 65, 0 ]);
            au.extend_from_slice(&[ 1, 2, 3, 4, 5, 6, 7, 8 ]);
            if data_size != 0xffffffff {
                // audio data contains extra bytes, which should not be part of sample data
                au.extend_from_slice(&[ 0x80, 0x81, 0x82, 0x83, 0x84, 0x85 ]);
            }
            let mut cursor = Cursor::new(&au);
            read_dummy_data(&mut cursor, 3)?;
            let mut reader = AuReader::new(&mut cursor)?;
            let _ = reader.read_info()?;
            reader.seek(4)?;
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(5)));
            assert_eq!(reader.read_sample().expect("sample read error"), Some(Sample::I8(6)));

        }
        Ok(())
    }
}
