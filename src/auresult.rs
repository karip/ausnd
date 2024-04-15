use core::result;

/// Error values.
#[derive(Debug)]
pub enum AuError {
    /// Unrecognized format.
    UnrecognizedFormat,
    /// The stream header has an invalid data offset and can't be read.
    InvalidAudioDataOffset,
    /// The operation is not supported for the current encoding.
    Unsupported,
    /// Invalid read state.
    InvalidReadState,
    /// Invalid write state.
    InvalidWriteState,
    /// Invalid parameter.
    InvalidParameter,
    /// Invalid sample format.
    InvalidSampleFormat,
    /// Seek error.
    SeekError,
    /// Standard IO error.
    StdIoError(std::io::Error),
}

impl From<std::io::Error> for AuError {
    fn from(e: std::io::Error) -> Self {
        AuError::StdIoError(e)
    }
}

/// Library Result type.
pub type AuResult<T> = result::Result<T, AuError>;
