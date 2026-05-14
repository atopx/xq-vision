use image::RgbImage;
use ort::session::Session;
use ort::value::TensorRef;

use crate::config::ModelSource;
use crate::config::SessionConfig;
use crate::error::Result;
use crate::error::XqVisionError;
use crate::fast_path::argmax_f32;
use crate::geometry::Mat3;
use crate::geometry::affine_transform;
use crate::geometry::apply_mat3;
use crate::geometry::perspective_transform;
use crate::image_ops::TensorScratch;
use crate::image_ops::warp_rgb;
use crate::session::create_session;
use crate::types::BoardCorners;
use crate::types::BoardImage;
use crate::types::Point2f;
use crate::types::RectF;

pub const BOARD_CORNER_NAMES: [&str; 4] = ["A0", "A8", "J0", "J8"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardDetectorConfig {
    pub input_width: u32,
    pub input_height: u32,
    pub padding: f32,
}

impl Default for BoardDetectorConfig {
    fn default() -> Self { Self { input_width: 256, input_height: 256, padding: 1.25 } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoardDetection {
    pub corners: BoardCorners,
    pub scores: [f32; 4],
}

pub struct BoardDetector {
    session: Session,
    input_name: String,
    output_names: [String; 2],
    config: BoardDetectorConfig,
    tensor: TensorScratch,
}

impl BoardDetector {
    pub fn new(model: ModelSource) -> Result<Self> {
        Self::with_config(model, BoardDetectorConfig::default(), SessionConfig::default())
    }

    pub fn with_config(model: ModelSource, config: BoardDetectorConfig, session_config: SessionConfig) -> Result<Self> {
        let session = create_session(&model, &session_config)?;
        let input_name = session
            .inputs()
            .first()
            .ok_or(XqVisionError::OutputShape { model: "board detector", expected: "one input", actual: vec![] })?
            .name()
            .to_string();
        if session.outputs().len() < 2 {
            return Err(XqVisionError::OutputShape {
                model: "board detector",
                expected: "two simcc outputs",
                actual: vec![session.outputs().len()],
            });
        }
        let output_names = [session.outputs()[0].name().to_string(), session.outputs()[1].name().to_string()];
        let tensor = TensorScratch::with_capacity((config.input_width as usize) * (config.input_height as usize));
        Ok(Self { session, input_name, output_names, config, tensor })
    }

    #[must_use]
    pub fn config(&self) -> BoardDetectorConfig { self.config }

    pub fn detect(&mut self, image: &RgbImage) -> Result<BoardDetection> {
        self.detect_in_rect(image, RectF::from_image(image))
    }

    pub fn detect_in_rect(&mut self, image: &RgbImage, rect: RectF) -> Result<BoardDetection> {
        rect.validate()?;
        let (center, scale) = self.preprocess(image, rect)?;
        let input = TensorRef::from_array_view(self.tensor.view()?)?;
        let outputs = self.session.run(ort::inputs![self.input_name.as_str() => input])?;

        let x_view = outputs[self.output_names[0].as_str()].try_extract_array::<f32>()?;
        let y_view = outputs[self.output_names[1].as_str()].try_extract_array::<f32>()?;
        let x_shape = x_view.shape().to_vec();
        let y_shape = y_view.shape().to_vec();
        let x =
            x_view.as_slice_memory_order().ok_or(XqVisionError::NonContiguousOutput { model: "board detector x" })?;
        let y =
            y_view.as_slice_memory_order().ok_or(XqVisionError::NonContiguousOutput { model: "board detector y" })?;
        let config = self.config;
        let (norm, scores) = decode_simcc(config, x, &x_shape, y, &y_shape)?;
        let corners = map_to_original(config, norm, center, scale)?;
        Ok(BoardDetection { corners, scores })
    }

    fn preprocess(&mut self, image: &RgbImage, rect: RectF) -> Result<(Point2f, Point2f)> {
        let (center, scale) = self.bbox_center_scale(rect);
        let matrix = self.build_warp(center, scale, false)?;
        let warped = warp_rgb(image, &matrix, self.config.input_width, self.config.input_height)?;
        self.tensor.normalize_image(&warped);
        Ok((center, scale))
    }

    fn bbox_center_scale(&self, rect: RectF) -> (Point2f, Point2f) {
        (
            Point2f::new((rect.x_min + rect.x_max) * 0.5, (rect.y_min + rect.y_max) * 0.5),
            Point2f::new(rect.width() * self.config.padding, rect.height() * self.config.padding),
        )
    }

    fn build_warp(&self, center: Point2f, scale: Point2f, inverse: bool) -> Result<Mat3> {
        let width = self.config.input_width as f32;
        let height = self.config.input_height as f32;
        let aspect = width / height;
        let fixed = if scale.x > scale.y * aspect {
            Point2f::new(scale.x, scale.x / aspect)
        } else {
            Point2f::new(scale.y * aspect, scale.y)
        };
        Self::warp_matrix(center, fixed, width, height, inverse)
    }

    fn warp_matrix(center: Point2f, scale: Point2f, out_width: f32, out_height: f32, inverse: bool) -> Result<Mat3> {
        let src_dir = Point2f::new(scale.x * -0.5, 0.0);
        let dst_dir = Point2f::new(out_width * -0.5, 0.0);
        let src0 = center;
        let src1 = Point2f::new(center.x + src_dir.x, center.y + src_dir.y);
        let src2 = third_point(src0, src1);
        let dst0 = Point2f::new(out_width * 0.5, out_height * 0.5);
        let dst1 = Point2f::new(dst0.x + dst_dir.x, dst0.y + dst_dir.y);
        let dst2 = third_point(dst0, dst1);
        let src = [src0, src1, src2];
        let dst = [dst0, dst1, dst2];
        if inverse { affine_transform(&dst, &src) } else { affine_transform(&src, &dst) }
    }
}

fn decode_simcc(
    config: BoardDetectorConfig, simcc_x: &[f32], x_shape: &[usize], simcc_y: &[f32], y_shape: &[usize],
) -> Result<([Point2f; 4], [f32; 4])> {
    validate_simcc_shape("board detector x", x_shape, config.input_width as usize * 2)?;
    validate_simcc_shape("board detector y", y_shape, config.input_height as usize * 2)?;

    let x_dim = x_shape[2];
    let y_dim = y_shape[2];
    let mut keypoints = [Point2f::new(0.0, 0.0); 4];
    let mut scores = [0.0_f32; 4];
    for i in 0..4 {
        let (x_index, x_score) = argmax_f32(&simcc_x[i * x_dim..(i + 1) * x_dim]);
        let (y_index, y_score) = argmax_f32(&simcc_y[i * y_dim..(i + 1) * y_dim]);
        keypoints[i] = Point2f::new(
            x_index as f32 / (config.input_width as f32 * 2.0),
            y_index as f32 / (config.input_height as f32 * 2.0),
        );
        scores[i] = x_score * y_score;
    }
    Ok((keypoints, scores))
}

fn map_to_original(
    config: BoardDetectorConfig, keypoints: [Point2f; 4], center: Point2f, scale: Point2f,
) -> Result<BoardCorners> {
    let helper = BoardDetectorConfigHelper(config);
    let inverse = helper.build_warp(center, scale, true)?;
    let width = config.input_width as f32;
    let height = config.input_height as f32;
    let mut mapped = [Point2f::new(0.0, 0.0); 4];
    for (dst, point) in mapped.iter_mut().zip(keypoints) {
        let model_point = Point2f::new(point.x * width, point.y * height);
        *dst = apply_mat3(&inverse, model_point)
            .ok_or(XqVisionError::InvalidGeometry("homogeneous denominator is zero"))?;
    }
    Ok(BoardCorners::new(mapped[0], mapped[1], mapped[2], mapped[3]))
}

struct BoardDetectorConfigHelper(BoardDetectorConfig);

impl BoardDetectorConfigHelper {
    fn build_warp(&self, center: Point2f, scale: Point2f, inverse: bool) -> Result<Mat3> {
        let width = self.0.input_width as f32;
        let height = self.0.input_height as f32;
        let aspect = width / height;
        let fixed = if scale.x > scale.y * aspect {
            Point2f::new(scale.x, scale.x / aspect)
        } else {
            Point2f::new(scale.y * aspect, scale.y)
        };
        BoardDetector::warp_matrix(center, fixed, width, height, inverse)
    }
}

pub fn warp_board(image: &RgbImage, corners: BoardCorners) -> Result<BoardImage> {
    const DST_WIDTH: u32 = 450;
    const DST_HEIGHT: u32 = 500;
    const PADDING: f32 = 50.0;

    corners.validate()?;
    let src = corners.as_points();
    let dst = [
        Point2f::new(PADDING, PADDING),
        Point2f::new(DST_WIDTH as f32 - PADDING, PADDING),
        Point2f::new(PADDING, DST_HEIGHT as f32 - PADDING),
        Point2f::new(DST_WIDTH as f32 - PADDING, DST_HEIGHT as f32 - PADDING),
    ];
    let matrix = perspective_transform(&src, &dst)?;
    Ok(BoardImage::new(warp_rgb(image, &matrix, DST_WIDTH, DST_HEIGHT)?))
}

fn third_point(a: Point2f, b: Point2f) -> Point2f {
    let dir = Point2f::new(a.x - b.x, a.y - b.y);
    Point2f::new(b.x - dir.y, b.y + dir.x)
}

fn validate_simcc_shape(model: &'static str, shape: &[usize], expected_dim: usize) -> Result<()> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 4 || shape[2] != expected_dim {
        return Err(XqVisionError::OutputShape { model, expected: "[1, 4, 2 * input_size]", actual: shape.to_vec() });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn warp_board_returns_canonical_size() -> Result<()> {
        let image = RgbImage::from_pixel(100, 100, Rgb([30, 40, 50]));
        let corners = BoardCorners::new(
            Point2f::new(10.0, 10.0),
            Point2f::new(90.0, 10.0),
            Point2f::new(10.0, 90.0),
            Point2f::new(90.0, 90.0),
        );
        let board = warp_board(&image, corners)?;
        assert_eq!(board.width(), 450);
        assert_eq!(board.height(), 500);
        Ok(())
    }

    #[test]
    fn simcc_shape_validation_rejects_wrong_keypoint_count() {
        assert!(matches!(
            validate_simcc_shape("board_model", &[1, 3, 512], 512),
            Err(XqVisionError::OutputShape { .. })
        ));
    }

    #[test]
    fn full_image_rect_uses_width_then_height() {
        let image = RgbImage::new(320, 240);
        let rect = RectF::from_image(&image);
        assert_eq!(rect.width(), 320.0);
        assert_eq!(rect.height(), 240.0);
    }
}
