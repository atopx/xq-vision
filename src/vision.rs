use image::RgbImage;

use crate::board::BoardDetector;
use crate::board::BoardDetectorConfig;
use crate::board::warp_board;
use crate::config::GraphOptimization;
use crate::config::ModelSource;
use crate::config::SessionConfig;
use crate::error::Result;
use crate::error::XqVisionError;
use crate::pieces::PieceRecognizer;
use crate::pieces::PieceRecognizerConfig;
use crate::session::ProviderFailure;
use crate::types::RecognitionResult;

pub struct XqVision {
    board_detector: BoardDetector,
    piece_recognizer: PieceRecognizer,
}

impl XqVision {
    #[must_use]
    pub fn builder() -> XqVisionBuilder { XqVisionBuilder::default() }

    pub fn recognize(&mut self, image: &RgbImage) -> Result<RecognitionResult> {
        let board_detection = self.board_detector.detect(image)?;
        let board = warp_board(image, board_detection.corners)?;
        let pieces = self.piece_recognizer.recognize(&board)?;
        let side_to_move = pieces.infer_side_to_move()?;
        Ok(RecognitionResult::new(board_detection.corners, board_detection.scores, board, pieces, side_to_move))
    }

    #[must_use]
    pub fn board_detector(&self) -> &BoardDetector { &self.board_detector }

    #[must_use]
    pub fn board_detector_mut(&mut self) -> &mut BoardDetector { &mut self.board_detector }

    #[must_use]
    pub fn piece_recognizer(&self) -> &PieceRecognizer { &self.piece_recognizer }

    #[must_use]
    pub fn piece_recognizer_mut(&mut self) -> &mut PieceRecognizer { &mut self.piece_recognizer }
}

#[derive(Debug, Default, Clone)]
pub struct XqVisionBuilder {
    board_model: Option<ModelSource>,
    piece_model: Option<ModelSource>,
    board_config: BoardDetectorConfig,
    piece_config: PieceRecognizerConfig,
    session_config: SessionConfig,
}

impl XqVisionBuilder {
    #[must_use]
    pub fn board_model(mut self, model: impl Into<ModelSource>) -> Self {
        self.board_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn piece_model(mut self, model: impl Into<ModelSource>) -> Self {
        self.piece_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn board_config(mut self, config: BoardDetectorConfig) -> Self {
        self.board_config = config;
        self
    }

    #[must_use]
    pub fn piece_config(mut self, config: PieceRecognizerConfig) -> Self {
        self.piece_config = config;
        self
    }

    #[must_use]
    pub fn session_config(&self) -> &SessionConfig { &self.session_config }

    #[must_use]
    pub fn with_session_config(mut self, config: SessionConfig) -> Self {
        self.session_config = config;
        self
    }

    #[must_use]
    pub fn provider_failure(mut self, failure: ProviderFailure) -> Self {
        self.session_config = self.session_config.with_provider_failure(failure);
        self
    }

    #[must_use]
    pub fn graph_optimization(mut self, level: GraphOptimization) -> Self {
        self.session_config = self.session_config.with_graph_optimization(level);
        self
    }

    #[must_use]
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.session_config = self.session_config.with_intra_threads(threads);
        self
    }

    #[must_use]
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.session_config = self.session_config.with_inter_threads(threads);
        self
    }

    #[must_use]
    pub fn parallel_execution(mut self, enabled: bool) -> Self {
        self.session_config = self.session_config.with_parallel_execution(enabled);
        self
    }

    pub fn build(self) -> Result<XqVision> {
        let board_model = self.board_model.ok_or(XqVisionError::MissingModel { role: "board" })?;
        let piece_model = self.piece_model.ok_or(XqVisionError::MissingModel { role: "piece" })?;
        let board_detector = BoardDetector::with_config(board_model, self.board_config, self.session_config.clone())?;
        let piece_recognizer = PieceRecognizer::with_config(piece_model, self.piece_config, self.session_config)?;
        Ok(XqVision { board_detector, piece_recognizer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_board_model() {
        let result = XqVision::builder().piece_model(ModelSource::memory([1_u8])).build();
        assert!(matches!(result, Err(XqVisionError::MissingModel { role: "board" })));
    }

    #[test]
    fn builder_requires_piece_model() {
        let result = XqVision::builder().board_model(ModelSource::memory([1_u8])).build();
        assert!(matches!(result, Err(XqVisionError::MissingModel { role: "piece" })));
    }
}
