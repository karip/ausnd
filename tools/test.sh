
set -e # stop on errors

# ensure clippy gives no errors or warnings
cargo clippy -- -D warnings

# ensure documentation can be built
cargo doc

# run all tests
cargo test

echo "---t1"
ls -la
echo "---t2"
ls -la toisto-au-test-suite
echo "---t3"
ls -la ../target
echo "---t4"
ls -la ../target/debug
echo "---t5"
ls -la ../target/debug/examples
echo "---t6"

# build the toisto tester
cargo build --example ausnd-aureader-toisto

# run the toisto test suite
echo
echo 'Toisto AU test suite results:'
cd toisto-au-test-suite
python3 toisto-runner.py -c -v --override-list ../toisto-ausnd-override-list.json ../target/debug/examples/ausnd-aureader-toisto

echo
echo "--- All tests OK."
