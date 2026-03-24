use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};

/// Page-aligned buffer for direct I/O.
///
/// Uses Rust's global allocator (`std::alloc`) with an explicit alignment,
/// which works on macOS, Linux, and Windows without POSIX-specific APIs.
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `len` bytes aligned to `alignment` (must be a power of 2, ≥ 1).
    pub fn new(len: usize, alignment: usize) -> std::io::Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                layout: Layout::new::<u8>(),
            });
        }
        let layout = Layout::from_size_align(len, alignment)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::from(std::io::ErrorKind::OutOfMemory));
        }
        Ok(Self { ptr, len, layout })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { dealloc(self.ptr, self.layout) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let buf = AlignedBuffer::new(4096, 4096).unwrap();
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.as_ptr() as usize % 4096, 0);
    }

    #[test]
    fn test_zero_size() {
        let buf = AlignedBuffer::new(0, 4096).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_read_write() {
        let mut buf = AlignedBuffer::new(256, 4096).unwrap();
        buf[0] = 0xAA;
        buf[255] = 0xBB;
        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[255], 0xBB);
    }
}
