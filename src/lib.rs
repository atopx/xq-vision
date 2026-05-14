//! High-performance Chinese chessboard vision primitives and ONNX inference.
//!
//! The crate exposes a compact end-to-end API through [`XqVision`] while keeping
//! the board detector, piece recognizer, and board warp operation reusable for
//! callers that need finer control.
//!
//! ```no_run
//! use xq_vision::{ModelSource, XqVision};
//!
//! # fn main() -> xq_vision::Result<()> {
//! let mut vision = XqVision::builder()
//!     .board_model(ModelSource::file("models/board.onnx"))
//!     .piece_model(ModelSource::file("models/piece.onnx"))
//!     .build()?;
//!
//! let image = image::open("board.jpg")?.to_rgb8();
//! let result = vision.recognize(&image)?;
//! println!("{}", result.to_fen());
//! # Ok(())
//! # }
//! ```
//!
//! # Concurrency
//!
//! [`XqVision`] (along with [`BoardDetector`] and [`PieceRecognizer`]) is `Send`
//! but **intentionally not `Sync`**. `recognize` takes `&mut self` because it
//! reuses an internal tensor scratch buffer between calls, so an instance is
//! owned by exactly one task at a time.
//!
//! **Recommended async model** — give each worker its own `XqVision`:
//!
//! ```no_run
//! # use xq_vision::{ModelSource, XqVision};
//! # async fn run() -> xq_vision::Result<()> {
//! // Share model bytes cheaply via Arc; each task builds its own session.
//! let board_bytes: std::sync::Arc<[u8]> = std::fs::read("models/board.onnx")?.into();
//! let piece_bytes: std::sync::Arc<[u8]> = std::fs::read("models/piece.onnx")?.into();
//!
//! let mut vision = XqVision::builder()
//!     .board_model(ModelSource::Memory(board_bytes.clone()))
//!     .piece_model(ModelSource::Memory(piece_bytes.clone()))
//!     .build()?;
//! // ...use `vision` from a single task...
//! # let _ = &mut vision;
//! # Ok(())
//! # }
//! ```
//!
//! If a single instance must be shared, wrap it in `tokio::sync::Mutex<XqVision>`
//! or `std::sync::Mutex<XqVision>` — inference will be serialized.

mod board;
mod config;
mod error;
mod fast_path;
mod geometry;
mod image_ops;
mod pieces;
mod session;
mod types;
mod vision;

pub use board::BOARD_CORNER_NAMES;
pub use board::BoardDetection;
pub use board::BoardDetector;
pub use board::BoardDetectorConfig;
pub use board::warp_board;
pub use config::GraphOptimization;
pub use config::ModelSource;
pub use config::SessionConfig;
pub use error::Result;
pub use error::XqVisionError;
pub use pieces::PIECE_SHORT;
pub use pieces::PieceRecognition;
pub use pieces::PieceRecognizer;
pub use pieces::PieceRecognizerConfig;
pub use pieces::PieceSnapshot;
pub use session::ExecutionProvider;
pub use session::ProviderFailure;
pub use types::BoardCoord;
pub use types::BoardCorners;
pub use types::BoardImage;
pub use types::CellPrediction;
pub use types::PieceKind;
pub use types::Point2f;
pub use types::RecognitionResult;
pub use types::RectF;
pub use types::SideToMove;
pub use vision::XqVision;
pub use vision::XqVisionBuilder;

#[cfg(feature = "bench-support")]
#[doc(hidden)]
pub mod bench_support {
    use image::RgbImage;

    use crate::Result;
    use crate::fast_path::normalize_rgb_to_chw;
    use crate::pieces::PieceRecognition;
    use crate::pieces::PieceRecognizer;
    use crate::types::BOARD_CELLS;
    use crate::types::PIECE_CLASSES;

    #[must_use]
    pub fn normalize_for_bench(image: &RgbImage) -> Vec<f32> {
        let mut out = Vec::new();
        normalize_rgb_to_chw(image.as_raw(), image.width() as usize, image.height() as usize, &mut out);
        out
    }

    pub fn decode_logits_for_bench(logits: &[f32]) -> Result<PieceRecognition> {
        PieceRecognizer::decode_logits(logits)
    }

    #[must_use]
    pub fn zero_logits_for_bench() -> Vec<f32> { vec![0.0; BOARD_CELLS * PIECE_CLASSES] }
}

const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<XqVision>();
    assert_send::<BoardDetector>();
    assert_send::<PieceRecognizer>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_builder_default_providers_end_with_cpu() {
        let builder = XqVision::builder()
            .board_model(ModelSource::memory([1_u8, 2, 3]))
            .piece_model(ModelSource::memory([4_u8, 5, 6]));
        let providers = builder.session_config().execution_providers();
        assert_eq!(providers.last(), Some(&ExecutionProvider::Cpu), "Cpu must always be the final fallback provider");
    }
}
