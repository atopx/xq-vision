use nalgebra::Matrix3;
use nalgebra::SMatrix;
use nalgebra::SVector;

use crate::error::Result;
use crate::error::XqVisionError;
use crate::types::Point2f;

pub(crate) type Mat3 = [[f32; 3]; 3];

type Mat6 = SMatrix<f32, 6, 6>;
type Vec6 = SVector<f32, 6>;
type Mat8 = SMatrix<f32, 8, 8>;
type Vec8 = SVector<f32, 8>;

pub(crate) fn affine_transform(src: &[Point2f; 3], dst: &[Point2f; 3]) -> Result<Mat3> {
    let mut a = Mat6::zeros();
    let mut b = Vec6::zeros();
    for i in 0..3 {
        a[(i, 0)] = src[i].x;
        a[(i, 1)] = src[i].y;
        a[(i, 2)] = 1.0;
        a[(i + 3, 3)] = src[i].x;
        a[(i + 3, 4)] = src[i].y;
        a[(i + 3, 5)] = 1.0;
        b[i] = dst[i].x;
        b[i + 3] = dst[i].y;
    }
    let solved = a.lu().solve(&b).ok_or(XqVisionError::SingularTransform("affine"))?;
    Ok([[solved[0], solved[1], solved[2]], [solved[3], solved[4], solved[5]], [0.0, 0.0, 1.0]])
}

pub(crate) fn perspective_transform(src: &[Point2f; 4], dst: &[Point2f; 4]) -> Result<Mat3> {
    let mut a = Mat8::zeros();
    let mut b = Vec8::zeros();
    for i in 0..4 {
        let (x, y) = (src[i].x, src[i].y);
        let (u, v) = (dst[i].x, dst[i].y);
        a[(2 * i, 0)] = x;
        a[(2 * i, 1)] = y;
        a[(2 * i, 2)] = 1.0;
        a[(2 * i, 6)] = -x * u;
        a[(2 * i, 7)] = -y * u;
        b[2 * i] = u;
        a[(2 * i + 1, 3)] = x;
        a[(2 * i + 1, 4)] = y;
        a[(2 * i + 1, 5)] = 1.0;
        a[(2 * i + 1, 6)] = -x * v;
        a[(2 * i + 1, 7)] = -y * v;
        b[2 * i + 1] = v;
    }
    let solved = a.lu().solve(&b).ok_or(XqVisionError::SingularTransform("perspective"))?;
    Ok([[solved[0], solved[1], solved[2]], [solved[3], solved[4], solved[5]], [solved[6], solved[7], 1.0]])
}

pub(crate) fn invert(m: &Mat3) -> Result<Mat3> {
    let matrix = Matrix3::<f32>::new(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
    let inv = matrix.try_inverse().ok_or(XqVisionError::SingularTransform("matrix"))?;
    Ok([
        [inv[(0, 0)], inv[(0, 1)], inv[(0, 2)]],
        [inv[(1, 0)], inv[(1, 1)], inv[(1, 2)]],
        [inv[(2, 0)], inv[(2, 1)], inv[(2, 2)]],
    ])
}

#[inline]
pub(crate) fn apply_mat3(m: &Mat3, point: Point2f) -> Option<Point2f> {
    let denom = m[2][0] * point.x + m[2][1] * point.y + m[2][2];
    if denom.abs() <= f32::EPSILON {
        return None;
    }
    Some(Point2f::new(
        (m[0][0] * point.x + m[0][1] * point.y + m[0][2]) / denom,
        (m[1][0] * point.x + m[1][1] * point.y + m[1][2]) / denom,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_identity_maps_points() -> Result<()> {
        let src = [Point2f::new(0.0, 0.0), Point2f::new(10.0, 0.0), Point2f::new(0.0, 10.0)];
        let matrix = affine_transform(&src, &src)?;
        let mapped = apply_mat3(&matrix, Point2f::new(4.0, 5.0)).unwrap();
        assert!((mapped.x - 4.0).abs() < 1e-5);
        assert!((mapped.y - 5.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn singular_affine_returns_error() {
        let src = [Point2f::new(1.0, 1.0), Point2f::new(1.0, 1.0), Point2f::new(1.0, 1.0)];
        let dst = [Point2f::new(0.0, 0.0), Point2f::new(10.0, 0.0), Point2f::new(0.0, 10.0)];
        assert!(matches!(affine_transform(&src, &dst), Err(XqVisionError::SingularTransform("affine"))));
    }

    #[test]
    fn perspective_identity_inverts() -> Result<()> {
        let points =
            [Point2f::new(0.0, 0.0), Point2f::new(10.0, 0.0), Point2f::new(0.0, 10.0), Point2f::new(10.0, 10.0)];
        let matrix = perspective_transform(&points, &points)?;
        let inverse = invert(&matrix)?;
        let mapped = apply_mat3(&inverse, Point2f::new(3.0, 7.0)).unwrap();
        assert!((mapped.x - 3.0).abs() < 1e-5);
        assert!((mapped.y - 7.0).abs() < 1e-5);
        Ok(())
    }
}
