use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::io::aligned_buffer::AlignedBuffer;
use crate::io::compat::{self, NativeFd};

/// A request to read a region from disk.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub offset: u64,
    pub length: usize,
    pub tag: String,
}

/// Statistics for a completed read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadStats {
    pub bytes_read: u64,
    pub duration_us: u64,
    pub throughput_mbps: f64,
}

/// A completed read with its data.
pub struct ReadResult {
    pub request: ReadRequest,
    pub data: AlignedBuffer,
    pub stats: ReadStats,
}

/// Double-buffered async disk reader with cache bypass.
///
/// Spawns a background task that reads into alternating aligned buffers.
/// Consumer receives completed reads via an mpsc channel.
pub struct NvmePrefetcher {
    _file_path: PathBuf,
    request_tx: mpsc::UnboundedSender<ReadRequest>,
    result_rx: mpsc::UnboundedReceiver<ReadResult>,
    _handle: tokio::task::JoinHandle<()>,
}

impl NvmePrefetcher {
    /// Open a file for prefetching with OS cache bypass.
    /// `buffer_size` is the size of each double-buffer (e.g., 4 MiB).
    pub fn open(path: impl AsRef<Path>, buffer_size: usize) -> std::io::Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<ReadRequest>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<ReadResult>();

        let path_clone = file_path.clone();

        let handle = tokio::task::spawn_blocking(move || {
            let fd: NativeFd = match compat::open_direct_fd(&path_clone) {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!(
                        "NvmePrefetcher: failed to open {}: {e}",
                        path_clone.display()
                    );
                    return;
                }
            };

            while let Some(req) = request_rx.blocking_recv() {
                let mut buf = match AlignedBuffer::new(buffer_size.max(req.length), 4096) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("NvmePrefetcher: alloc failed: {e}");
                        continue;
                    }
                };

                let start = Instant::now();
                let mut total_read = 0usize;
                let to_read = req.length;

                while total_read < to_read {
                    let n = compat::read_at_fd(
                        fd,
                        buf[total_read..].as_mut_ptr(),
                        to_read - total_read,
                        req.offset + total_read as u64,
                    );
                    if n <= 0 {
                        break;
                    }
                    total_read += n as usize;
                }

                let duration = start.elapsed();
                let duration_us = duration.as_micros() as u64;
                let throughput_mbps = if duration_us > 0 {
                    total_read as f64 / duration.as_secs_f64() / 1e6
                } else {
                    0.0
                };

                let result = ReadResult {
                    request: req,
                    data: buf,
                    stats: ReadStats {
                        bytes_read: total_read as u64,
                        duration_us,
                        throughput_mbps,
                    },
                };

                if result_tx.send(result).is_err() {
                    break; // Consumer dropped
                }
            }

            compat::close_fd(fd);
        });

        Ok(Self {
            _file_path: file_path,
            request_tx,
            result_rx,
            _handle: handle,
        })
    }

    /// Enqueue a read request. Returns immediately.
    pub fn submit(&self, request: ReadRequest) -> anyhow::Result<()> {
        self.request_tx
            .send(request)
            .map_err(|_| anyhow::anyhow!("Prefetcher background thread has stopped"))
    }

    /// Receive the next completed read. Blocks until available.
    pub async fn recv(&mut self) -> Option<ReadResult> {
        self.result_rx.recv().await
    }
}
