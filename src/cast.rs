
/// Casts i8 to u8, mapping negative values to 128..255.
#[allow(clippy::cast_sign_loss)] // mapping to positive values is expected
pub const fn i8_as_u8(value: i8) -> u8 {
    value as u8
}

/// Casts u8 to i8, mapping 128..255 to negative values.
#[allow(clippy::cast_possible_wrap)] // mapping to negative values is expected
pub const fn u8_as_i8(value: u8) -> i8 {
    value as i8
}
