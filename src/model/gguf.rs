use std::collections::BTreeMap;
use std::io::{self, Read, Seek};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("Invalid GGUF magic: got 0x{0:08X}")]
    InvalidMagic(u32),
    #[error("Unsupported GGUF version: {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid string: {0}")]
    InvalidString(#[from] std::string::FromUtf8Error),
    #[error("Unknown value type: {0}")]
    UnknownValueType(u32),
    #[error("Unknown tensor type: {0}")]
    UnknownTensorType(u32),
}

/// Parsed GGUF file (header only — tensor data is not loaded).
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset in the file where tensor data begins.
    pub data_offset: u64,
}

/// A metadata value in the GGUF key-value store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            Self::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }
}

/// GGML quantization / data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    Unknown(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Bytes per element (for non-quantized types) or bytes per block.
    /// Returns (block_size_elements, block_size_bytes).
    pub fn block_size(&self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::BF16 => (1, 2),
            Self::F64 => (1, 8),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::IQ2XXS => (256, 66),
            Self::IQ2XS => (256, 74),
            Self::IQ3XXS => (256, 98),
            Self::IQ1S => (256, 50),
            Self::IQ4NL => (32, 18),
            Self::IQ3S => (256, 110),
            Self::IQ2S => (256, 82),
            Self::IQ4XS => (32, 18),
            Self::IQ1M => (256, 56),
            Self::Unknown(_) => (1, 1), // fallback
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GgmlType,
    /// Byte offset relative to data_offset.
    pub offset: u64,
    /// Computed total size in bytes.
    pub size_bytes: u64,
    /// Parsed layer index from tensor name (e.g., "blk.5.xxx" → Some(5)).
    pub layer_index: Option<u32>,
}

impl GgufFile {
    /// Open and parse a GGUF file's header (does not load tensor data).
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path)?;
        let mut reader = io::BufReader::new(file);
        Self::parse(&mut reader)
    }

    /// Parse GGUF header from any reader.
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self, GgufError> {
        // Magic
        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Version
        let version = reader.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Counts
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

        // Metadata
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(reader)?;
            let value = read_value(reader)?;
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let dtype_raw = reader.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw);
            let offset = reader.read_u64::<LittleEndian>()?;

            let num_elements: u64 = dimensions.iter().product();
            let (block_elems, block_bytes) = dtype.block_size();
            let size_bytes = (num_elements / block_elems) * block_bytes;

            let layer_index = parse_layer_index(&name);

            tensors.push(TensorInfo {
                name,
                dimensions,
                dtype,
                offset,
                size_bytes,
                layer_index,
            });
        }

        // Data offset: aligned to GGUF_DEFAULT_ALIGNMENT (32 bytes)
        let current_pos = reader.stream_position()?;
        let alignment = 32u64;
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Total bytes of all tensor data.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes).sum()
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a string metadata value, trying both `general.X` and `X` keys.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .or_else(|| {
                self.metadata
                    .get(&format!("general.{key}"))
                    .and_then(|v| v.as_str())
            })
    }

    /// Get a u32 metadata value, trying architecture-prefixed keys.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32()).or_else(|| {
            // Try with architecture prefix
            let arch = self.get_string("general.architecture")?;
            self.metadata
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.as_u32())
        })
    }
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}

fn read_value<R: Read>(reader: &mut R) -> Result<GgufValue, GgufError> {
    let vtype = reader.read_u32::<LittleEndian>()?;
    read_value_of_type(reader, vtype)
}

fn read_value_of_type<R: Read>(reader: &mut R, vtype: u32) -> Result<GgufValue, GgufError> {
    match vtype {
        0 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        1 => Ok(GgufValue::Int8(reader.read_i8()?)),
        2 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => {
            let val = reader.read_u8()?;
            Ok(GgufValue::Bool(val != 0))
        }
        8 => {
            let s = read_string(reader)?;
            Ok(GgufValue::String(s))
        }
        9 => {
            // Array: element_type (u32) + count (u64) + elements
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_value_of_type(reader, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
        other => Err(GgufError::UnknownValueType(other)),
    }
}

fn parse_layer_index(name: &str) -> Option<u32> {
    // Match patterns like "blk.5.xxx" or "layers.5.xxx"
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 && (parts[0] == "blk" || parts[0] == "layers") {
        parts[1].parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_gguf_header(buf: &mut Vec<u8>, version: u32, tensor_count: u64, kv_count: u64) {
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
        write_string(buf, value);
    }

    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type = uint32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn write_tensor_info(
        buf: &mut Vec<u8>,
        name: &str,
        dims: &[u64],
        dtype: u32,
        offset: u64,
    ) {
        write_string(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&dtype.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 3, 1, 2);

        // Metadata
        write_kv_string(&mut buf, "general.architecture", "llama");
        write_kv_u32(&mut buf, "llama.block_count", 32);

        // One tensor: F32, shape [4096, 4096]
        write_tensor_info(&mut buf, "blk.0.attn_q.weight", &[4096, 4096], 0, 0);

        let mut cursor = Cursor::new(buf);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.metadata.len(), 2);
        assert_eq!(
            gguf.get_string("general.architecture"),
            Some("llama")
        );
        assert_eq!(gguf.tensors.len(), 1);
        assert_eq!(gguf.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(gguf.tensors[0].dtype, GgmlType::F32);
        assert_eq!(gguf.tensors[0].size_bytes, 4096 * 4096 * 4); // F32 = 4 bytes
        assert_eq!(gguf.tensors[0].layer_index, Some(0));
    }

    #[test]
    fn test_parse_quantized_tensor() {
        let mut buf = Vec::new();
        write_gguf_header(&mut buf, 3, 1, 0);

        // Q4_K tensor: block_size=(256, 144)
        write_tensor_info(&mut buf, "blk.5.ffn_gate.weight", &[4096, 4096], 12, 0);

        let mut cursor = Cursor::new(buf);
        let gguf = GgufFile::parse(&mut cursor).unwrap();

        assert_eq!(gguf.tensors[0].dtype, GgmlType::Q4K);
        // 4096*4096 / 256 * 144 = 9,437,184
        assert_eq!(gguf.tensors[0].size_bytes, 4096 * 4096 / 256 * 144);
        assert_eq!(gguf.tensors[0].layer_index, Some(5));
    }
}
