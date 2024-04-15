/*!

Test command to read AU files and outputs json for Toisto AU test suite.

Example run:

    cargo run --example ausnd-aureader-toisto filename.au

*/
use std::fs::File;
use std::io::BufReader;
use std::env;

#[path = "../tests/shared/jsonhelper.rs"]
mod jsonhelper;

fn main() {
    let num_args = env::args().count();
    if num_args != 2 {
        println!("Usage: ausnd-aureader-toisto filename.au");
        return;
    }

    // open file for reading
    let filename = env::args().nth(1).expect("Cannot get filename");
    let mut rd = BufReader::new(File::open(filename)
        .expect("Failed to open file for reading"));
    let mut reader = ausnd::AuReader::new(&mut rd)
        .expect("Failed to read the AU file");

    // print json

    let json = jsonhelper::jsonify(&mut reader).expect("Failed to jsonify");
    println!("{}", json);
}
