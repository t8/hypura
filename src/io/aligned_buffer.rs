use std::ops::{Deref, DerefMut};

/// Page-aligned buffer for direct I/O (F_NOCACHE).
/// Allocated via `posix_memalign`, freed via `libc::free`.
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate `len` bytes aligned to `alignment` (must be a power of 2, typically 4096).
    pub fn new(len: usize, alignment: usize) -> std::io::Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let ret = unsafe { libc::posix_memalign(&mut ptr, alignment, len) };
        if ret != 0 {
            return Err(std::io::Error::from_raw_os_error(ret));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the buffer's contents.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
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
            unsafe { libc::free(self.ptr as *mut libc::c_void) }
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
