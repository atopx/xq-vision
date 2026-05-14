use thiserror::Error;

use crate::session::ExecutionProvider;

pub type Result<T> = std::result::Result<T, XqVisionError>;

#[derive(Debug, Error)]
pub enum XqVisionError {
    #[error("missing {role} model")]
    MissingModel { role: &'static str },

    #[error("invalid image size {width}x{height}; expected at least {min_width}x{min_height}")]
    ImageTooSmall { width: u32, height: u32, min_width: u32, min_height: u32 },

    #[error("invalid geometry: {0}")]
    InvalidGeometry(&'static str),

    #[error("singular {0} transform")]
    SingularTransform(&'static str),

    #[error("{model} output shape mismatch; expected {expected}, got {actual:?}")]
    OutputShape { model: &'static str, expected: &'static str, actual: Vec<usize> },

    #[error("{model} output is not contiguous in memory")]
    NonContiguousOutput { model: &'static str },

    #[error("invalid piece class index {0}")]
    InvalidPieceIndex(u8),

    #[error("recognition result is missing one or both kings")]
    MissingKings,

    #[error("execution provider {provider:?} requires an enabled Cargo feature or supported platform")]
    UnsupportedProvider { provider: ExecutionProvider },

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("onnx runtime error: {0}")]
    Ort(#[from] ort::Error),
}
