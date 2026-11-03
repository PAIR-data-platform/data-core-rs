//! Frame-accurate range decoding of LeRobot video files with FFmpeg.

use std::path::Path;
use std::sync::OnceLock;

use ffmpeg_next as ffmpeg;
use image::ImageFormat;

use data_core::{DataCoreError, Result};

/// Decodes ranges of video frames using the system FFmpeg libraries.
///
/// Supports all codecs available in the installed libav build, including
/// H.264 and AV1. Each call opens the file, seeks to the keyframe at or
/// before the requested start, and decodes forward, discarding frames whose
/// PTS is before the target. Frame `k` is assumed to have PTS
/// `k / fps` seconds (in the stream time base), which holds for the
/// constant-frame-rate videos LeRobot writes.
///
/// FFmpeg is not fork-safe: call these functions only in the process that
/// will use the results, not in a forked worker that inherited FFmpeg state.
pub struct VideoDecoder;

fn init_ffmpeg_with_quiet_logs() -> Result<()> {
    static FFMPEG_SETUP: OnceLock<std::result::Result<(), String>> = OnceLock::new();

    let setup = FFMPEG_SETUP.get_or_init(|| {
        // Suppress codec/library info chatter such as:
        // "[libdav1d @ ...] libdav1d 0.9.2"
        // Keep ffmpeg errors visible.
        unsafe {
            ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_ERROR);
        }
        ffmpeg::init().map_err(|e| format!("ffmpeg init failed: {e}"))
    });

    match setup {
        Ok(()) => Ok(()),
        Err(msg) => Err(DataCoreError::Other(msg.clone())),
    }
}

impl VideoDecoder {
    /// Decode exactly `count` frames starting at `start_frame` (0-based index within
    /// the video file). Uses ffmpeg seeking to jump to the approximate position, then
    /// skips any pre-target frames by comparing PTS values.
    ///
    /// Returns up to `count` JPEG-encoded frames in display order (fewer if
    /// the video ends first; callers check the length). Frames are converted
    /// to RGB24 and then JPEG-encoded with the `image` crate's default
    /// quality.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, has no video stream,
    /// or fails to seek, decode, convert, or encode.
    pub fn decode_range(
        path: &Path,
        start_frame: i64,
        count: usize,
        fps: f64,
    ) -> Result<Vec<Vec<u8>>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        init_ffmpeg_with_quiet_logs()?;

        let path_buf = path.to_path_buf();
        let mut ictx = ffmpeg::format::input(&path_buf).map_err(|e| {
            DataCoreError::Other(format!("Failed to open video {}: {e}", path.display()))
        })?;

        let stream_index = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| DataCoreError::Other(format!("No video stream in {}", path.display())))?
            .index();

        // Capture the stream timebase before building the decoder (borrow rules).
        let (tb_num, tb_den) = {
            let stream = ictx.stream(stream_index).ok_or_else(|| {
                DataCoreError::Other("Stream not found after index lookup".to_string())
            })?;
            let tb = stream.time_base();
            (tb.0 as i64, tb.1 as i64)
        };

        // Build the decoder from stream parameters (ffmpeg-next 7.x API).
        let mut decoder = {
            let stream = ictx
                .stream(stream_index)
                .ok_or_else(|| DataCoreError::Other("Stream not found".to_string()))?;
            let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| DataCoreError::Other(format!("Codec context: {e}")))?;
            context
                .decoder()
                .video()
                .map_err(|e| DataCoreError::Other(format!("Video decoder: {e}")))?
        };

        let width = decoder.width();
        let height = decoder.height();
        let src_format = decoder.format();

        // Software scaler: source pixel format → RGB24.
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            src_format,
            width,
            height,
            ffmpeg::format::pixel::Pixel::RGB24,
            width,
            height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .map_err(|e| DataCoreError::Other(format!("Scaler init: {e}")))?;

        // PTS of the target start frame in the stream's timebase.
        // pts_per_frame = tb_den / (fps * tb_num)
        // For AV1 at 10 fps with timebase 1/12800: pts_per_frame = 1280.
        let pts_per_frame = tb_den as f64 / (fps * tb_num as f64);
        let start_pts = (start_frame as f64 * pts_per_frame).round() as i64;

        // Seek to just before the target frame (in microseconds = AV_TIME_BASE units).
        if start_frame > 0 {
            let seek_us = ((start_frame as f64 / fps) * 1_000_000.0) as i64;
            // Seek to or before the target timestamp.
            ictx.seek(seek_us, ..seek_us).map_err(|e| {
                DataCoreError::Other(format!("Seek to frame {start_frame} failed: {e}"))
            })?;
        }

        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(count);
        let mut decoded = ffmpeg::util::frame::video::Video::empty();
        let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();

        'outer: for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index {
                continue;
            }
            if frames.len() >= count {
                break;
            }

            decoder
                .send_packet(&packet)
                .map_err(|e| DataCoreError::Other(format!("send_packet: {e}")))?;

            loop {
                match decoder.receive_frame(&mut decoded) {
                    Ok(()) => {}
                    Err(ffmpeg::Error::Other {
                        errno: ffmpeg::util::error::EAGAIN,
                    })
                    | Err(ffmpeg::Error::Eof) => break,
                    Err(e) => return Err(DataCoreError::Other(format!("receive_frame: {e}"))),
                }

                // After seeking the demuxer may emit frames from the preceding keyframe.
                // Skip any frame whose PTS precedes our target.
                if let Some(pts) = decoded.pts() {
                    if pts < start_pts {
                        continue;
                    }
                }

                if frames.len() >= count {
                    break 'outer;
                }

                Self::encode_frame(
                    &mut scaler,
                    &decoded,
                    &mut rgb_frame,
                    &mut frames,
                    width,
                    height,
                )?;
            }
        }

        // Flush the decoder to collect any buffered frames.
        decoder
            .send_eof()
            .map_err(|e| DataCoreError::Other(format!("send_eof: {e}")))?;
        loop {
            if frames.len() >= count {
                break;
            }
            match decoder.receive_frame(&mut decoded) {
                Ok(()) => {}
                Err(ffmpeg::Error::Other {
                    errno: ffmpeg::util::error::EAGAIN,
                })
                | Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(DataCoreError::Other(format!("receive_frame (flush): {e}"))),
            }
            if let Some(pts) = decoded.pts() {
                if pts < start_pts {
                    continue;
                }
            }
            Self::encode_frame(
                &mut scaler,
                &decoded,
                &mut rgb_frame,
                &mut frames,
                width,
                height,
            )?;
        }

        Ok(frames)
    }

    /// Decode exactly `count` frames starting at `start_frame` as raw RGB pixels.
    ///
    /// Same as `decode_range` but returns raw RGB bytes (H×W×3, row-major)
    /// instead of JPEG, together with the frame `(width, height)`.
    ///
    /// Avoids the JPEG encode/decode round trip, so pixels are exactly the
    /// decoder output. When `count` is 0, returns no frames and a size of
    /// `(0, 0)` without opening the file.
    ///
    /// # Errors
    ///
    /// Same as [`VideoDecoder::decode_range`].
    pub fn decode_range_raw(
        path: &Path,
        start_frame: i64,
        count: usize,
        fps: f64,
    ) -> Result<(Vec<Vec<u8>>, u32, u32)> {
        if count == 0 {
            return Ok((Vec::new(), 0, 0));
        }

        init_ffmpeg_with_quiet_logs()?;

        let path_buf = path.to_path_buf();
        let mut ictx = ffmpeg::format::input(&path_buf).map_err(|e| {
            DataCoreError::Other(format!("Failed to open video {}: {e}", path.display()))
        })?;

        let stream_index = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| DataCoreError::Other(format!("No video stream in {}", path.display())))?
            .index();

        let (tb_num, tb_den) = {
            let stream = ictx.stream(stream_index).ok_or_else(|| {
                DataCoreError::Other("Stream not found after index lookup".to_string())
            })?;
            let tb = stream.time_base();
            (tb.0 as i64, tb.1 as i64)
        };

        let mut decoder = {
            let stream = ictx
                .stream(stream_index)
                .ok_or_else(|| DataCoreError::Other("Stream not found".to_string()))?;
            let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
                .map_err(|e| DataCoreError::Other(format!("Codec context: {e}")))?;
            context
                .decoder()
                .video()
                .map_err(|e| DataCoreError::Other(format!("Video decoder: {e}")))?
        };

        let width = decoder.width();
        let height = decoder.height();
        let src_format = decoder.format();

        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            src_format,
            width,
            height,
            ffmpeg::format::pixel::Pixel::RGB24,
            width,
            height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .map_err(|e| DataCoreError::Other(format!("Scaler init: {e}")))?;

        let pts_per_frame = tb_den as f64 / (fps * tb_num as f64);
        let start_pts = (start_frame as f64 * pts_per_frame).round() as i64;

        if start_frame > 0 {
            let seek_us = ((start_frame as f64 / fps) * 1_000_000.0) as i64;
            ictx.seek(seek_us, ..seek_us).map_err(|e| {
                DataCoreError::Other(format!("Seek to frame {start_frame} failed: {e}"))
            })?;
        }

        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(count);
        let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
        let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();

        let collect_frame = |scaler: &mut ffmpeg::software::scaling::context::Context,
                             decoded: &ffmpeg::util::frame::video::Video,
                             rgb: &mut ffmpeg::util::frame::video::Video,
                             out: &mut Vec<Vec<u8>>|
         -> Result<()> {
            Self::extract_raw_pixels(scaler, decoded, rgb, out, width, height)
        };

        'outer: for (stream, packet) in ictx.packets() {
            if stream.index() != stream_index || frames.len() >= count {
                continue;
            }
            decoder
                .send_packet(&packet)
                .map_err(|e| DataCoreError::Other(format!("send_packet: {e}")))?;
            loop {
                match decoder.receive_frame(&mut decoded_frame) {
                    Ok(()) => {}
                    Err(ffmpeg::Error::Other {
                        errno: ffmpeg::util::error::EAGAIN,
                    })
                    | Err(ffmpeg::Error::Eof) => break,
                    Err(e) => return Err(DataCoreError::Other(format!("receive_frame: {e}"))),
                }
                if let Some(pts) = decoded_frame.pts() {
                    if pts < start_pts {
                        continue;
                    }
                }
                if frames.len() >= count {
                    break 'outer;
                }
                collect_frame(&mut scaler, &decoded_frame, &mut rgb_frame, &mut frames)?;
            }
        }

        // Flush
        decoder
            .send_eof()
            .map_err(|e| DataCoreError::Other(format!("send_eof: {e}")))?;
        loop {
            if frames.len() >= count {
                break;
            }
            match decoder.receive_frame(&mut decoded_frame) {
                Ok(()) => {}
                Err(ffmpeg::Error::Other {
                    errno: ffmpeg::util::error::EAGAIN,
                })
                | Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(DataCoreError::Other(format!("receive_frame (flush): {e}"))),
            }
            if let Some(pts) = decoded_frame.pts() {
                if pts < start_pts {
                    continue;
                }
            }
            collect_frame(&mut scaler, &decoded_frame, &mut rgb_frame, &mut frames)?;
        }

        Ok((frames, width, height))
    }

    /// Extract raw RGB pixels from a decoded frame (no JPEG encoding).
    fn extract_raw_pixels(
        scaler: &mut ffmpeg::software::scaling::context::Context,
        decoded: &ffmpeg::util::frame::video::Video,
        rgb_frame: &mut ffmpeg::util::frame::video::Video,
        frames: &mut Vec<Vec<u8>>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        scaler
            .run(decoded, rgb_frame)
            .map_err(|e| DataCoreError::Other(format!("scaler.run: {e}")))?;

        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data(0);
        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height as usize {
            let row_start = y * stride;
            pixels.extend_from_slice(&data[row_start..row_start + width as usize * 3]);
        }

        frames.push(pixels);
        Ok(())
    }

    /// Encode a single decoded frame to JPEG and push it into `frames`.
    fn encode_frame(
        scaler: &mut ffmpeg::software::scaling::context::Context,
        decoded: &ffmpeg::util::frame::video::Video,
        rgb_frame: &mut ffmpeg::util::frame::video::Video,
        frames: &mut Vec<Vec<u8>>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        scaler
            .run(decoded, rgb_frame)
            .map_err(|e| DataCoreError::Other(format!("scaler.run: {e}")))?;

        let stride = rgb_frame.stride(0);
        let data = rgb_frame.data(0);
        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height as usize {
            let row_start = y * stride;
            pixels.extend_from_slice(&data[row_start..row_start + width as usize * 3]);
        }

        let rgb_img = image::RgbImage::from_raw(width, height, pixels).ok_or_else(|| {
            DataCoreError::Other("Failed to construct RgbImage from decoded frame".to_string())
        })?;

        let mut jpeg_buf = Vec::new();
        image::DynamicImage::ImageRgb8(rgb_img)
            .write_to(&mut std::io::Cursor::new(&mut jpeg_buf), ImageFormat::Jpeg)
            .map_err(|e| DataCoreError::Other(format!("JPEG encode: {e}")))?;

        frames.push(jpeg_buf);
        Ok(())
    }
}
