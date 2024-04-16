# Ausnd

[![Cross-platform tests](https://github.com/karip/ausnd/actions/workflows/cross-test.yml/badge.svg)](https://github.com/karip/ausnd/actions/workflows/cross-test.yml)

Rust library to read and write
Sun/Next [AU audio format](https://en.wikipedia.org/wiki/Au_file_format).
Features:

 - read AU files
 - write AU files
 - reading infinite AU streams
 - no heap memory allocations
 - no unsafe code
 - no panicking
 - supports uncompressed integer and floating point samples
 - supports compression types: μ-law and A-law
 - supports audio streams larger than 4 gigabytes

Out of scope:

 - conversion between different sample formats (e.g., i16 to f32). There's
   [so many ways](http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html)
   to do the conversion that it's better that this crate doesn't do it.

## Usage

Reading an AU audio file:

```rust, no_run
let mut bufrd = std::io::BufReader::new(std::fs::File::open("test.au").expect("File error"));
let mut reader = ausnd::AuReader::new(&mut bufrd).expect("Read error");
let info = reader.read_info().expect("Invalid header");
for sample in reader.samples().expect("Can't read samples") {
    println!("Got sample {:?}", sample.expect("Sample error"));
}
```

Writing an AU audio file (with the default 2 channels, 16-bit signed integer samples,
sample rate 44100):

```rust, no_run
let mut bufwr = std::io::BufWriter::new(std::fs::File::create("test.au").expect("File error"));
let winfo = ausnd::AuWriteInfo::default();
let mut writer = ausnd::AuWriter::new(&mut bufwr, &winfo).expect("Write header error");
writer.write_samples_i16(&[ 1, 2, 3, 4 ]).expect("Write sample error");
writer.finalize().expect("Finalize error");
```

See [the ausnd API documentation](https://docs.rs/ausnd/) for details.

## Examples

A simple AU player using `ausnd::AuReader` and [tinyaudio](https://crates.io/crates/tinyaudio):

```sh
cd examples/ausnd-tinyaudio
cargo run filename.au
```

A simple audio processor for volume and noise effects using `AuStreamParser` and `AuWriter`:

```sh
cargo run --example ausnd-piper v0.8 n0.1 < input.au > out.au
```

The same audio processor piped from/to ffmpeg. ffmpeg converts mp3 to AU for ausnd-piper,
which writes AU for ffmpeg to convert it to mp3.

```sh
ffmpeg -i music.mp3 -f au - | cargo run --example ausnd-piper v1.5 n0.2 | ffmpeg -i - -y out.mp3
```

## Testing

[Toisto AU Test Suite](https://github.com/karip/toisto-au-test-suite) is a submodule and
needs to be fetched before running the integration tests.

```sh
cd ausnd
git submodule update --init
./tools/test.sh
```

The test should end with `--- All tests OK.`.

Performance testing:

```sh
cargo bench
```

There is a GitHub Action called "Cross-platform tests" (cross-test.yml), which automatically
runs `./tools/test.sh` for little-endian 64-bit x64_86 and big-endian 32-bit PowerPC.

## References

 - [Wikipedia: Au file format](https://en.wikipedia.org/wiki/Au_file_format)
 - [Oracle AU audio file format man page](https://docs.oracle.com/cd/E36784_01/html/E36882/au-4.html)
 - [Audio File Formats FAQ: File Formats (archived)](https://web.archive.org/web/20230223152815/https://sox.sourceforge.net/AudioFormats-11.html#ss11.2)
 - [NeXT/Sun soundfile format](http://soundfile.sapp.org/doc/NextFormat/)
 - [NeXT soundstruct.h](https://github.com/johnsonjh/NeXTDSP/blob/26d2b31a6fb4bc16d55ebe17824cd2d6f9edfc7b/sound-33/soundstruct.h#L4)
 - [SunOS audio_filehdr.h](https://github.com/Arquivotheca/SunOS-4.1.3/blob/413/demo/SOUND/multimedia/audio_filehdr.h)

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
