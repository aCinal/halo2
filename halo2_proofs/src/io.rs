//! I/O traits used by this crate.
//!
//! When the `std` feature is enabled this module simply re-exports from
//! [`std::io`].  Without `std` a minimal compatible implementation is
//! provided so that `&[u8]` and `Vec<u8>` can be used as proof buffers.

#[cfg(feature = "std")]
pub use std::io::{Error, ErrorKind, Read, Result, Write};

#[cfg(not(feature = "std"))]
pub use self::nostd::{Error, ErrorKind, Read, Result, Write};

#[cfg(not(feature = "std"))]
mod nostd {
    use alloc::vec::Vec;

    /// A specialised [`Result`](core::result::Result) for I/O operations.
    pub type Result<T> = core::result::Result<T, Error>;

    /// The error type for I/O operations.
    #[derive(Debug)]
    pub struct Error {
        kind: ErrorKind,
        desc: &'static str,
    }

    impl Error {
        /// Creates a new I/O error from a known kind and a static description.
        pub fn new(kind: ErrorKind, desc: &'static str) -> Self {
            Error { kind, desc }
        }

        /// Returns the corresponding [`ErrorKind`] for this error.
        pub fn kind(&self) -> ErrorKind {
            self.kind
        }
    }

    impl core::fmt::Display for Error {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.desc)
        }
    }

    /// A list specifying general categories of I/O error.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ErrorKind {
        /// An error returned when an operation could not be completed because an
        /// "end of file" was reached prematurely.
        UnexpectedEof,
        /// Invalid input parameter was passed.
        InvalidInput,
        /// An error not covered by any other variant.
        Other,
    }

    /// The `Read` trait allows for reading bytes from a source.
    pub trait Read {
        /// Pull some bytes from this source into the specified buffer, returning
        /// how many bytes were read.
        fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

        /// Read the exact number of bytes required to fill `buf`.
        fn read_exact(&mut self, mut buf: &mut [u8]) -> Result<()> {
            while !buf.is_empty() {
                match self.read(buf) {
                    Ok(0) => {
                        return Err(Error::new(
                            ErrorKind::UnexpectedEof,
                            "failed to fill whole buffer",
                        ));
                    }
                    Ok(n) => {
                        let tmp = buf;
                        buf = &mut tmp[n..];
                    }
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }

    /// A trait for objects which are byte-oriented sinks.
    pub trait Write {
        /// Write a buffer into this writer, returning how many bytes were written.
        fn write(&mut self, buf: &[u8]) -> Result<usize>;

        /// Flush this output stream, ensuring that all buffered contents reach
        /// their destination.
        fn flush(&mut self) -> Result<()>;

        /// Attempts to write an entire buffer into this writer.
        fn write_all(&mut self, mut buf: &[u8]) -> Result<()> {
            while !buf.is_empty() {
                match self.write(buf) {
                    Ok(0) => {
                        return Err(Error::new(
                            ErrorKind::Other,
                            "failed to write whole buffer",
                        ));
                    }
                    Ok(n) => buf = &buf[n..],
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }

    impl Read for &[u8] {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            let len = core::cmp::min(buf.len(), self.len());
            let (a, b) = self.split_at(len);
            buf[..len].copy_from_slice(a);
            *self = b;
            Ok(len)
        }
    }

    impl<R: Read> Read for &mut R {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            (*self).read(buf)
        }
    }

    impl Write for Vec<u8> {
        fn write(&mut self, buf: &[u8]) -> Result<usize> {
            self.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }
}
