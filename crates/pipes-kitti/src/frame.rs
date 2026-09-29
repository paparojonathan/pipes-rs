//! Arrow `RecordBatch` for one cam0 frame. The decoded pixel `Vec<u8>` becomes
//! the `pixels` LargeBinary buffer without a copy (`Buffer::from_vec`), and its
//! address is the frame's `storage_id`, re-derived at every stage as the
//! standing zero-copy proof.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, LargeBinaryArray, RecordBatch, StringArray, UInt32Array};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use arrow::error::ArrowError;

/// Errors from building a cam0 batch.
#[derive(Debug)]
pub enum FrameError {
    /// `pixels.len()` is not `width * height * 3`.
    BadLength {
        /// Bytes the dimensions imply (`width * height * 3`).
        expected: usize,
        /// Bytes actually supplied.
        got: usize,
    },
    /// Arrow rejected the array or the batch.
    Arrow(ArrowError),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::BadLength { expected, got } => {
                write!(
                    f,
                    "pixel buffer length {got} != width * height * 3 = {expected}"
                )
            }
            FrameError::Arrow(e) => write!(f, "arrow: {e}"),
        }
    }
}

impl std::error::Error for FrameError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FrameError::BadLength { .. } => None,
            FrameError::Arrow(e) => Some(e),
        }
    }
}

impl From<ArrowError> for FrameError {
    fn from(e: ArrowError) -> Self {
        FrameError::Arrow(e)
    }
}

/// Value of the `pixel_format` column for 8-bit interleaved RGB.
pub const PIXEL_FORMAT_RGB8: &str = "rgb8";

/// `width u32, height u32, stride u32, pixel_format Utf8, pixels LargeBinary`,
/// all non-null. Call once per run and pass the `Arc` around.
pub fn cam0_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("width", DataType::UInt32, false),
        Field::new("height", DataType::UInt32, false),
        Field::new("stride", DataType::UInt32, false),
        Field::new("pixel_format", DataType::Utf8, false),
        Field::new("pixels", DataType::LargeBinary, false),
    ]))
}

/// Wraps `pixels` as a 1-row batch without copying it. Returns the batch and
/// the `storage_id` (address of the pixel buffer).
pub fn build_cam0_batch(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    schema: &Arc<Schema>,
) -> Result<(RecordBatch, usize), FrameError> {
    let len = pixels.len();
    let expected = width as usize * height as usize * 3;
    if len != expected {
        return Err(FrameError::BadLength { expected, got: len });
    }
    let values = Buffer::from_vec(pixels);
    let storage_id = values.as_ptr() as usize;
    let offsets = OffsetBuffer::<i64>::new(ScalarBuffer::<i64>::from(vec![0i64, len as i64]));
    let pixels_arr = LargeBinaryArray::try_new(offsets, values, None)?;
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt32Array::from(vec![width])),
        Arc::new(UInt32Array::from(vec![height])),
        Arc::new(UInt32Array::from(vec![width * 3])),
        Arc::new(StringArray::from(vec![PIXEL_FORMAT_RGB8])),
        Arc::new(pixels_arr),
    ];
    Ok((
        RecordBatch::try_new(Arc::clone(schema), columns)?,
        storage_id,
    ))
}

/// The shared pixel buffer (offset 0) of a cam0 batch.
pub fn cam0_pixels(batch: &RecordBatch) -> Option<&Buffer> {
    Some(
        batch
            .column_by_name("pixels")?
            .as_binary_opt::<i64>()?
            .values(),
    )
}

/// Re-derived storage id for the 3-stage proof.
pub fn cam0_storage_id(batch: &RecordBatch) -> Option<usize> {
    cam0_pixels(batch).map(|b| b.as_ptr() as usize)
}

/// `(width, height)` of a cam0 batch.
pub fn cam0_dims(batch: &RecordBatch) -> Option<(u32, u32)> {
    let w = batch
        .column_by_name("width")?
        .as_primitive_opt::<UInt32Type>()?;
    let h = batch
        .column_by_name("height")?
        .as_primitive_opt::<UInt32Type>()?;
    if w.is_empty() || h.is_empty() {
        return None;
    }
    Some((w.value(0), h.value(0)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn g2a_pixels_ptr_equals_storage_id() {
        let (w, h) = (64u32, 8u32);
        let pixels = vec![7u8; (w * h * 3) as usize];
        let p_vec = pixels.as_ptr() as usize;
        let schema = cam0_schema();
        let (batch, storage_id) = build_cam0_batch(pixels, w, h, &schema).unwrap();
        assert_eq!(storage_id, p_vec, "Buffer::from_vec copied");
        let arr = batch.column_by_name("pixels").unwrap().as_binary::<i64>();
        assert_eq!(arr.value(0).as_ptr() as usize, storage_id);
        assert_eq!(arr.value(0).len(), (w * h * 3) as usize);
        assert_eq!(arr.values().as_ptr() as usize, storage_id);
        assert_eq!(cam0_storage_id(&batch), Some(storage_id));
        assert_eq!(cam0_dims(&batch), Some((w, h)));
    }

    #[test]
    fn g2b_batch_alloc_small_and_resolution_invariant() {
        use pipes_core::alloc::{bytes_alloc, set_stage_slot, SLOT_TEST};
        let schema = cam0_schema(); // hoisted: not counted
        set_stage_slot(SLOT_TEST);
        let measure = |w: u32, h: u32| -> u64 {
            let pixels = vec![0u8; (w * h * 3) as usize]; // the pixel allocation happens BEFORE the window
            let before = bytes_alloc(SLOT_TEST);
            let (batch, _) = build_cam0_batch(pixels, w, h, &schema).unwrap();
            let after = bytes_alloc(SLOT_TEST);
            drop(batch);
            after - before
        };
        let _warm = measure(16, 16); // absorbs any first-call one-off
        let small = measure(64, 8);
        let full = measure(1242, 375);
        println!("g2b: batch construction allocated small={small} B full={full} B");
        assert!(small < 2048, "batch construction allocated {small} B");
        assert_eq!(
            small, full,
            "allocation depends on resolution: {small} vs {full}"
        );
    }

    #[test]
    fn bad_length_is_rejected() {
        let schema = cam0_schema();
        let err = build_cam0_batch(vec![0u8; 10], 2, 2, &schema).unwrap_err();
        assert!(matches!(
            err,
            FrameError::BadLength {
                expected: 12,
                got: 10
            }
        ));
    }
}
