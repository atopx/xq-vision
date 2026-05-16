use image::RgbImage;
use ort::session::Session;
use ort::value::TensorRef;

use crate::config::ModelSource;
use crate::config::SessionConfig;
use crate::error::Result;
use crate::error::XqVisionError;
use crate::fast_path::argmax_f32;
use crate::image_ops::TensorScratch;
use crate::image_ops::resize_center_crop_rgb;
use crate::session::create_session;
use crate::types::BOARD_CELLS;
use crate::types::BOARD_FILES;
use crate::types::BOARD_RANKS;
use crate::types::BoardCoord;
use crate::types::BoardImage;
use crate::types::CellPrediction;
use crate::types::PIECE_CLASSES;
use crate::types::PieceKind;
use crate::types::Side;

pub const PIECE_SHORT: [char; PIECE_CLASSES] =
    ['.', 'x', 'K', 'A', 'B', 'N', 'R', 'C', 'P', 'k', 'a', 'b', 'n', 'r', 'c', 'p'];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PieceRecognizerConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub crop_width: u32,
    pub crop_height: u32,
}

impl Default for PieceRecognizerConfig {
    fn default() -> Self { Self { input_width: 280, input_height: 315, crop_width: 400, crop_height: 450 } }
}

/// Combined snapshot of all three per-cell views (indexes, FEN-short chars,
/// confidences) produced by [`PieceRecognition::snapshot`] in a single pass.
#[derive(Debug, Clone)]
pub struct PieceSnapshot {
    /// Per-cell piece index. Mirrors [`PieceRecognition::indexes`].
    pub indexes: [[u8; BOARD_FILES]; BOARD_RANKS],
    /// Per-cell FEN-style short character. Mirrors [`PieceRecognition::shorts`].
    pub shorts: [[char; BOARD_FILES]; BOARD_RANKS],
    /// Per-cell confidence in [0, 1]. Mirrors [`PieceRecognition::confidence`].
    pub confidence: [[f32; BOARD_FILES]; BOARD_RANKS],
}

#[derive(Debug, Clone)]
pub struct PieceRecognition {
    cells: [[CellPrediction; BOARD_FILES]; BOARD_RANKS],
}

impl PieceRecognition {
    #[must_use]
    pub fn new(cells: [[CellPrediction; BOARD_FILES]; BOARD_RANKS]) -> Self { Self { cells } }

    #[must_use]
    pub fn cells(&self) -> &[[CellPrediction; BOARD_FILES]; BOARD_RANKS] { &self.cells }

    #[must_use]
    pub fn indexes(&self) -> [[u8; BOARD_FILES]; BOARD_RANKS] {
        let mut out = [[0_u8; BOARD_FILES]; BOARD_RANKS];
        for (rank, row) in self.cells.iter().enumerate() {
            for (file, cell) in row.iter().enumerate() {
                out[rank][file] = cell.piece.index();
            }
        }
        out
    }

    #[must_use]
    pub fn shorts(&self) -> [[char; BOARD_FILES]; BOARD_RANKS] {
        let mut out = [['.'; BOARD_FILES]; BOARD_RANKS];
        for (rank, row) in self.cells.iter().enumerate() {
            for (file, cell) in row.iter().enumerate() {
                out[rank][file] = cell.piece.short();
            }
        }
        out
    }

    #[must_use]
    pub fn confidence(&self) -> [[f32; BOARD_FILES]; BOARD_RANKS] {
        let mut out = [[0.0_f32; BOARD_FILES]; BOARD_RANKS];
        for (rank, row) in self.cells.iter().enumerate() {
            for (file, cell) in row.iter().enumerate() {
                out[rank][file] = cell.confidence;
            }
        }
        out
    }

    /// Single-pass projection of all three per-cell views (indexes, FEN-short
    /// chars, confidences). Prefer this over calling `indexes()`, `shorts()`,
    /// and `confidence()` separately when you need all three — it walks the
    /// 90-cell grid once instead of three times, improving cache locality.
    #[must_use]
    pub fn snapshot(&self) -> PieceSnapshot {
        let mut indexes = [[0_u8; BOARD_FILES]; BOARD_RANKS];
        let mut shorts = [['.'; BOARD_FILES]; BOARD_RANKS];
        let mut confidence = [[0.0_f32; BOARD_FILES]; BOARD_RANKS];
        for (rank, row) in self.cells.iter().enumerate() {
            for (file, cell) in row.iter().enumerate() {
                indexes[rank][file] = cell.piece.index();
                shorts[rank][file] = cell.piece.short();
                confidence[rank][file] = cell.confidence;
            }
        }
        PieceSnapshot { indexes, shorts, confidence }
    }

    pub fn user_side(&self) -> Result<Side> {
        let mut user_side = None;
        for row in self.cells.iter().skip(BOARD_RANKS / 2) {
            for cell in row {
                let side = match cell.piece {
                    PieceKind::RedKing => Side::Red,
                    PieceKind::BlackKing => Side::Black,
                    _ => continue,
                };
                match user_side {
                    Some(existing) if existing != side => return Err(XqVisionError::InvalidBoard),
                    Some(_) => {}
                    None => user_side = Some(side),
                }
            }
        }
        user_side.ok_or(XqVisionError::InvalidBoard)
    }

    #[must_use]
    pub fn to_fen_placement(&self) -> String {
        let mut output = String::with_capacity(BOARD_CELLS + BOARD_RANKS - 1);
        for (rank, row) in self.cells.iter().enumerate() {
            if rank > 0 {
                output.push('/');
            }
            let mut empty = 0usize;
            for cell in row {
                if cell.piece == PieceKind::Empty {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    output.push((b'0' + empty as u8) as char);
                    empty = 0;
                }
                output.push(cell.piece.short());
            }
            if empty > 0 {
                output.push((b'0' + empty as u8) as char);
            }
        }
        output
    }

    #[must_use]
    pub fn to_fen(&self, side: Side) -> String { format!("{} {}", self.to_fen_placement(), side.fen_char()) }
}

pub struct PieceRecognizer {
    session: Session,
    input_name: String,
    output_name: String,
    config: PieceRecognizerConfig,
    tensor: TensorScratch,
}

impl PieceRecognizer {
    pub fn new(model: ModelSource) -> Result<Self> {
        Self::with_config(model, PieceRecognizerConfig::default(), SessionConfig::default())
    }

    pub fn with_config(
        model: ModelSource, config: PieceRecognizerConfig, session_config: SessionConfig,
    ) -> Result<Self> {
        let session = create_session(&model, &session_config)?;
        let input_name = session
            .inputs()
            .first()
            .ok_or(XqVisionError::OutputShape { model: "piece recognizer", expected: "one input", actual: vec![] })?
            .name()
            .to_string();
        let output_name = session
            .outputs()
            .first()
            .ok_or(XqVisionError::OutputShape { model: "piece recognizer", expected: "one output", actual: vec![] })?
            .name()
            .to_string();
        let tensor = TensorScratch::with_capacity((config.input_width as usize) * (config.input_height as usize));
        Ok(Self { session, input_name, output_name, config, tensor })
    }

    #[must_use]
    pub fn config(&self) -> PieceRecognizerConfig { self.config }

    pub fn recognize(&mut self, board: &BoardImage) -> Result<PieceRecognition> {
        self.recognize_image(board.as_image())
    }

    pub fn recognize_image(&mut self, image: &RgbImage) -> Result<PieceRecognition> {
        let prepared = resize_center_crop_rgb(
            image,
            self.config.crop_width,
            self.config.crop_height,
            self.config.input_width,
            self.config.input_height,
        )?;
        self.tensor.normalize_image(&prepared);
        let input = TensorRef::from_array_view(self.tensor.view()?)?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => input])?;
        let logits_view = outputs[self.output_name.as_str()].try_extract_array::<f32>()?;
        let shape = logits_view.shape().to_vec();
        validate_logits_shape(&shape)?;
        let logits = logits_view
            .as_slice_memory_order()
            .ok_or(XqVisionError::NonContiguousOutput { model: "piece recognizer" })?;
        Self::decode_logits(logits)
    }

    pub(crate) fn decode_logits(logits: &[f32]) -> Result<PieceRecognition> {
        if logits.len() != BOARD_CELLS * PIECE_CLASSES {
            return Err(XqVisionError::OutputShape {
                model: "piece recognizer",
                expected: "[1, 90, 16]",
                actual: vec![logits.len()],
            });
        }

        let empty = CellPrediction::new(BoardCoord::new(0, 0), PieceKind::Empty, 0.0);
        let mut cells = [[empty; BOARD_FILES]; BOARD_RANKS];
        for cell_index in 0..BOARD_CELLS {
            let class_start = cell_index * PIECE_CLASSES;
            let (piece_index, confidence) = argmax_f32(&logits[class_start..class_start + PIECE_CLASSES]);
            let rank = cell_index / BOARD_FILES;
            let file = cell_index % BOARD_FILES;
            let piece = PieceKind::from_index(piece_index as u8)?;
            cells[rank][file] = CellPrediction::new(BoardCoord::new(rank, file), piece, confidence);
        }
        Ok(PieceRecognition::new(cells))
    }
}

fn validate_logits_shape(shape: &[usize]) -> Result<()> {
    if shape != [1, BOARD_CELLS, PIECE_CLASSES] {
        return Err(XqVisionError::OutputShape {
            model: "piece recognizer",
            expected: "[1, 90, 16]",
            actual: shape.to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_logits_selects_best_piece_per_cell() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::RedKing.index() as usize] = 9.0;
        let black_cell = BOARD_CELLS - 1;
        logits[black_cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 0.0;
        logits[black_cell * PIECE_CLASSES + PieceKind::BlackKing.index() as usize] = 8.0;

        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert_eq!(recognition.cells()[0][0].piece, PieceKind::RedKing);
        assert_eq!(recognition.cells()[9][8].piece, PieceKind::BlackKing);
        assert_eq!(recognition.user_side()?, Side::Black);
        Ok(())
    }

    #[test]
    fn user_side_uses_lower_half_red_king() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::BlackKing.index() as usize] = 8.0;
        let red_cell = BOARD_CELLS - 1;
        logits[red_cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 0.0;
        logits[red_cell * PIECE_CLASSES + PieceKind::RedKing.index() as usize] = 9.0;

        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert_eq!(recognition.user_side()?, Side::Red);
        Ok(())
    }

    #[test]
    fn user_side_rejects_missing_lower_half_king() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::BlackKing.index() as usize] = 8.0;

        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert!(matches!(recognition.user_side(), Err(XqVisionError::InvalidBoard)));
        Ok(())
    }

    #[test]
    fn user_side_rejects_conflicting_lower_half_kings() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        let red_cell = BOARD_FILES * (BOARD_RANKS / 2);
        logits[red_cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 0.0;
        logits[red_cell * PIECE_CLASSES + PieceKind::RedKing.index() as usize] = 9.0;
        let black_cell = red_cell + 1;
        logits[black_cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 0.0;
        logits[black_cell * PIECE_CLASSES + PieceKind::BlackKing.index() as usize] = 8.0;

        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert!(matches!(recognition.user_side(), Err(XqVisionError::InvalidBoard)));
        Ok(())
    }

    #[test]
    fn fen_placement_compresses_empty_cells() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::BlackRook.index() as usize] = 5.0;
        logits[8 * PIECE_CLASSES] = 0.0;
        logits[8 * PIECE_CLASSES + PieceKind::BlackRook.index() as usize] = 5.0;
        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert!(recognition.to_fen_placement().starts_with("r7r/"));
        Ok(())
    }

    #[test]
    fn fen_includes_user_side() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::BlackRook.index() as usize] = 5.0;

        let recognition = PieceRecognizer::decode_logits(&logits)?;
        assert_eq!(recognition.to_fen(Side::Red), "r8/9/9/9/9/9/9/9/9/9 w");
        assert_eq!(recognition.to_fen(Side::Black), "r8/9/9/9/9/9/9/9/9/9 b");
        Ok(())
    }

    #[test]
    fn logits_shape_validation_rejects_wrong_shape() {
        assert!(matches!(validate_logits_shape(&[1, 89, 16]), Err(XqVisionError::OutputShape { .. })));
    }

    // Guard against drift between `snapshot` (single pass) and the individual
    // `indexes` / `shorts` / `confidence` getters.
    #[test]
    fn snapshot_matches_individual_getters() -> Result<()> {
        let mut logits = vec![0.0_f32; BOARD_CELLS * PIECE_CLASSES];
        for cell in 0..BOARD_CELLS {
            logits[cell * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 1.0;
        }
        logits[0] = 0.0;
        logits[PieceKind::RedKing.index() as usize] = 9.0;
        logits[5 * PIECE_CLASSES + PieceKind::Empty.index() as usize] = 0.0;
        logits[5 * PIECE_CLASSES + PieceKind::BlackRook.index() as usize] = 4.5;
        let recognition = PieceRecognizer::decode_logits(&logits)?;
        let snap = recognition.snapshot();
        assert_eq!(snap.indexes, recognition.indexes());
        assert_eq!(snap.shorts, recognition.shorts());
        assert_eq!(snap.confidence, recognition.confidence());
        Ok(())
    }
}
