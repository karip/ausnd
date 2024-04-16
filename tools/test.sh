
set -e # stop on errors

if [ ! -d "toisto-au-test-suite/tests" ]; then
    echo "toisto-au-test-suite is missing. Read README.md Testing section for instructions."
    exit 1
fi

# ensure clippy gives no errors or warnings
cargo clippy -- -D warnings

# ensure documentation can be built
cargo doc

# run tests - no capture if running under GitHub Actions
if [ "$GITHUB_ACTIONS" == "true" ]; then
    cargo test -- --nocapture
else
    cargo test
fi

# build the toisto tester
cargo build --example ausnd-aureader-toisto

# run the toisto test suite
echo
echo 'Toisto AU test suite results:'
cd toisto-au-test-suite

# target/debug for normal testing,
# x86_64-unknown-linux-gnu and powerpc-unknown-linux-gnu for GitHub Actions
if [ -e ../target/x86_64-unknown-linux-gnu/debug/examples/ausnd-aureader-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-ausnd-override-list.json ../target/x86_64-unknown-linux-gnu/debug/examples/ausnd-aureader-toisto
elif [ -e ../target/powerpc-unknown-linux-gnu/debug/examples/ausnd-aureader-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-ausnd-override-list.json ../target/powerpc-unknown-linux-gnu/debug/examples/ausnd-aureader-toisto
else
    python3 toisto-runner.py -c --override-list ../toisto-ausnd-override-list.json ../target/debug/examples/ausnd-aureader-toisto
fi

echo
echo "--- All tests OK."
