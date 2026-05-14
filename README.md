# xq-vision

`xq-vision` is a Rust library for Chinese chess (Xiangqi) board recognition from RGB images. It runs a two-stage ONNX pipeline:

1. `BoardDetector` predicts four board corners.
2. `warp_board` rectifies the board to a canonical `450x500` RGB image.
3. `PieceRecognizer` classifies all `10x9` cells into 16 piece classes.

The crate does not ship ONNX model weights. Applications provide model files or in-memory model bytes.

## Install

```toml
[dependencies]
xq-vision = "0.1"
```

Default features download/copy the ONNX Runtime binaries and enable the internal fast path:

```toml
xq-vision = { version = "0.1", default-features = true }
```

For fully managed runtime deployment, disable the default runtime download and opt into dynamic loading:

```toml
xq-vision = { version = "0.1", default-features = false, features = ["load-dynamic", "fast-path"] }
```

## Feature Matrix

| Feature | Purpose |
| --- | --- |
| `runtime-download` | Use `ort` binary download/copy support for easy local setup. Enabled by default. |
| `fast-path` | Enables internal unsafe pointer loops and runtime-detected SIMD argmax. Enabled by default. |
| `load-dynamic` | Use dynamically loaded ONNX Runtime libraries. |
| `coreml` | Enable Apple CoreML execution provider. |
| `cuda` | Enable NVIDIA CUDA execution provider. |
| `tensorrt` | Enable NVIDIA TensorRT execution provider. |
| `directml` | Enable Windows DirectML execution provider. |
| `openvino` | Enable Intel OpenVINO execution provider. |
| `xnnpack` | Enable XNNPACK execution provider. |

CPU execution is always available through ONNX Runtime. Execution providers are selected in order, and the default is CPU only.

## End-to-End Use

```rust
use xq_vision::{ExecutionProvider, ModelSource, XqVision};

fn main() -> xq_vision::Result<()> {
    let image = image::open("board.jpg")?.to_rgb8();
    let mut vision = XqVision::builder()
        .board_model(ModelSource::file("models/board.onnx"))
        .piece_model(ModelSource::file("models/piece.onnx"))
        .execution_providers([ExecutionProvider::Xnnpack, ExecutionProvider::Cpu])
        .build()?;

    let result = vision.recognize(&image)?;
    println!("{}", result.to_fen());
    Ok(())
}
```

## Advanced Components

```rust
use xq_vision::{BoardDetector, ModelSource, PieceRecognizer, warp_board};

fn main() -> xq_vision::Result<()> {
    let image = image::open("board.jpg")?.to_rgb8();
    let mut board_detector = BoardDetector::new(ModelSource::file("models/board.onnx"))?;
    let mut piece_recognizer = PieceRecognizer::new(ModelSource::file("models/piece.onnx"))?;

    let detection = board_detector.detect(&image)?;
    let board = warp_board(&image, detection.corners)?;
    let pieces = piece_recognizer.recognize(&board)?;
    let side_to_move = pieces.infer_side_to_move()?;
    println!("{}", pieces.to_fen(side_to_move));
    Ok(())
}
```

## Performance Policy

The public API is safe Rust. The default `fast-path` feature uses low-level unsafe code only inside internal image/tensor hot paths and keeps safe fallbacks plus consistency tests. ONNX sessions and scratch buffers are reused across calls, so reuse `XqVision`, `BoardDetector`, and `PieceRecognizer` instances instead of creating them per image.

Run local benchmarks with:

```bash
cargo bench --features bench-support
```

## Basic Example

The repository does not include model files in the published crate. In this checkout, real inference can be run against `./examples/test.jpg` and rendered into a local static HTML report:

```bash
BOARD_MODEL=./models/board.onnx PIECE_MODEL=./models/piece.onnx cargo run --example basic
```

The example writes `./examples/basic.html`. Open it directly in a browser to inspect the original image and the recognized board drawing.

## Release Checks

Run before publishing:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo test --all-features
cargo package
```
