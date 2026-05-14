use std::fmt;

use image::RgbImage;

use crate::error::Result;
use crate::error::XqVisionError;

pub const BOARD_RANKS: usize = 10;
pub const BOARD_FILES: usize = 9;
pub const BOARD_CELLS: usize = BOARD_RANKS * BOARD_FILES;
pub const PIECE_CLASSES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

impl Point2f {
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self { Self { x, y } }

    #[must_use]
    pub const fn to_array(self) -> [f32; 2] { [self.x, self.y] }
}

impl From<[f32; 2]> for Point2f {
    fn from(value: [f32; 2]) -> Self { Self::new(value[0], value[1]) }
}

impl From<Point2f> for [f32; 2] {
    fn from(value: Point2f) -> Self { value.to_array() }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectF {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl RectF {
    #[must_use]
    pub const fn new(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self { Self { x_min, y_min, x_max, y_max } }

    #[must_use]
    pub fn from_image(image: &RgbImage) -> Self { Self::new(0.0, 0.0, image.width() as f32, image.height() as f32) }

    #[must_use]
    pub fn width(self) -> f32 { self.x_max - self.x_min }

    #[must_use]
    pub fn height(self) -> f32 { self.y_max - self.y_min }

    pub(crate) fn validate(self) -> Result<()> {
        if self.width() <= 0.0 || self.height() <= 0.0 {
            return Err(XqVisionError::InvalidGeometry("bounding box must have positive size"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardCorners {
    pub top_left: Point2f,
    pub top_right: Point2f,
    pub bottom_left: Point2f,
    pub bottom_right: Point2f,
}

impl BoardCorners {
    #[must_use]
    pub const fn new(top_left: Point2f, top_right: Point2f, bottom_left: Point2f, bottom_right: Point2f) -> Self {
        Self { top_left, top_right, bottom_left, bottom_right }
    }

    #[must_use]
    pub fn as_points(self) -> [Point2f; 4] { [self.top_left, self.top_right, self.bottom_left, self.bottom_right] }

    #[must_use]
    pub fn as_arrays(self) -> [[f32; 2]; 4] { self.as_points().map(Point2f::to_array) }

    pub(crate) fn validate(self) -> Result<()> {
        let points = self.as_points();
        if points.iter().any(|point| !point.x.is_finite() || !point.y.is_finite()) {
            return Err(XqVisionError::InvalidGeometry("board corners must be finite"));
        }
        Ok(())
    }
}

impl From<[[f32; 2]; 4]> for BoardCorners {
    fn from(value: [[f32; 2]; 4]) -> Self {
        Self::new(value[0].into(), value[1].into(), value[2].into(), value[3].into())
    }
}

#[derive(Debug, Clone)]
pub struct BoardImage {
    image: RgbImage,
}

impl BoardImage {
    #[must_use]
    pub fn new(image: RgbImage) -> Self { Self { image } }

    #[must_use]
    pub fn as_image(&self) -> &RgbImage { &self.image }

    #[must_use]
    pub fn into_image(self) -> RgbImage { self.image }

    #[must_use]
    pub fn width(&self) -> u32 { self.image.width() }

    #[must_use]
    pub fn height(&self) -> u32 { self.image.height() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoardCoord {
    pub rank: usize,
    pub file: usize,
}

impl BoardCoord {
    #[must_use]
    pub const fn new(rank: usize, file: usize) -> Self { Self { rank, file } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KingPositions {
    pub red: BoardCoord,
    pub black: BoardCoord,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Empty = 0,
    Unknown = 1,
    RedKing = 2,
    RedAdvisor = 3,
    RedBishop = 4,
    RedKnight = 5,
    RedRook = 6,
    RedCannon = 7,
    RedPawn = 8,
    BlackKing = 9,
    BlackAdvisor = 10,
    BlackBishop = 11,
    BlackKnight = 12,
    BlackRook = 13,
    BlackCannon = 14,
    BlackPawn = 15,
}

impl PieceKind {
    pub const ALL: [Self; PIECE_CLASSES] = [
        Self::Empty,
        Self::Unknown,
        Self::RedKing,
        Self::RedAdvisor,
        Self::RedBishop,
        Self::RedKnight,
        Self::RedRook,
        Self::RedCannon,
        Self::RedPawn,
        Self::BlackKing,
        Self::BlackAdvisor,
        Self::BlackBishop,
        Self::BlackKnight,
        Self::BlackRook,
        Self::BlackCannon,
        Self::BlackPawn,
    ];

    #[must_use]
    pub const fn index(self) -> u8 { self as u8 }

    #[must_use]
    pub const fn short(self) -> char {
        match self {
            Self::Empty => '.',
            Self::Unknown => 'x',
            Self::RedKing => 'K',
            Self::RedAdvisor => 'A',
            Self::RedBishop => 'B',
            Self::RedKnight => 'N',
            Self::RedRook => 'R',
            Self::RedCannon => 'C',
            Self::RedPawn => 'P',
            Self::BlackKing => 'k',
            Self::BlackAdvisor => 'a',
            Self::BlackBishop => 'b',
            Self::BlackKnight => 'n',
            Self::BlackRook => 'r',
            Self::BlackCannon => 'c',
            Self::BlackPawn => 'p',
        }
    }

    pub fn from_index(index: u8) -> Result<Self> { Self::try_from(index) }
}

impl TryFrom<u8> for PieceKind {
    type Error = XqVisionError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Unknown),
            2 => Ok(Self::RedKing),
            3 => Ok(Self::RedAdvisor),
            4 => Ok(Self::RedBishop),
            5 => Ok(Self::RedKnight),
            6 => Ok(Self::RedRook),
            7 => Ok(Self::RedCannon),
            8 => Ok(Self::RedPawn),
            9 => Ok(Self::BlackKing),
            10 => Ok(Self::BlackAdvisor),
            11 => Ok(Self::BlackBishop),
            12 => Ok(Self::BlackKnight),
            13 => Ok(Self::BlackRook),
            14 => Ok(Self::BlackCannon),
            15 => Ok(Self::BlackPawn),
            other => Err(XqVisionError::InvalidPieceIndex(other)),
        }
    }
}

impl fmt::Display for PieceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.short().to_string()) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellPrediction {
    pub coord: BoardCoord,
    pub piece: PieceKind,
    pub confidence: f32,
}

impl CellPrediction {
    #[must_use]
    pub const fn new(coord: BoardCoord, piece: PieceKind, confidence: f32) -> Self { Self { coord, piece, confidence } }
}

#[derive(Debug, Clone)]
pub struct RecognitionResult {
    corners: BoardCorners,
    corner_scores: [f32; 4],
    board: BoardImage,
    pieces: crate::pieces::PieceRecognition,
}

impl RecognitionResult {
    #[must_use]
    pub(crate) fn new(
        corners: BoardCorners, corner_scores: [f32; 4], board: BoardImage, pieces: crate::pieces::PieceRecognition,
    ) -> Self {
        Self { corners, corner_scores, board, pieces }
    }

    #[must_use]
    pub fn corners(&self) -> BoardCorners { self.corners }

    #[must_use]
    pub fn corner_scores(&self) -> [f32; 4] { self.corner_scores }

    #[must_use]
    pub fn board(&self) -> &BoardImage { &self.board }

    #[must_use]
    pub fn pieces(&self) -> &crate::pieces::PieceRecognition { &self.pieces }

    #[must_use]
    pub fn into_parts(self) -> (BoardCorners, [f32; 4], BoardImage, crate::pieces::PieceRecognition) {
        (self.corners, self.corner_scores, self.board, self.pieces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_kind_round_trips_all_indices() -> Result<()> {
        for expected in PieceKind::ALL {
            assert_eq!(PieceKind::from_index(expected.index())?, expected);
        }
        Ok(())
    }

    #[test]
    fn invalid_piece_index_returns_error() {
        assert!(matches!(PieceKind::from_index(99), Err(XqVisionError::InvalidPieceIndex(99))));
    }
}
