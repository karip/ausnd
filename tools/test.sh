
set -e # stop on errors

# ensure clippy gives no errors or warnings
cargo clippy -- -D warnings

# ensure documentation can be built
cargo doc

# run all tests
cargo test

# run the toisto test suite
echo
echo 'Toisto AU test suite results:'
cd toisto-au-test-suite
python3 toisto-runner.py -c --override-list ../toisto-ausnd-override-list.json ../target/debug/examples/ausnd-aureader-toisto

echo
echo "--- All tests OK."
