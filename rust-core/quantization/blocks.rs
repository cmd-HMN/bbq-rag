pub const BASE_DIMS: usize = 128;
pub const BLOCK: usize = 32;
pub const BLOCK_SIZE: usize = BASE_DIMS / BLOCK;

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QF32 {
    pub data: [f32; BLOCK],
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QI8 {
    pub scale: f32,
    pub data: [i8; BLOCK],
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QBin {
    pub data: [u8; BLOCK],
}

// Default will be Float32
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QTYPE {
    Binary,
    Int8,
    #[default]
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QData {
    Binary([QBin; BLOCK_SIZE]),
    Int8([QI8; BLOCK_SIZE]),
    Float32([QF32; BLOCK_SIZE]),
}

impl Default for QData {
    fn default() -> Self {
        QData::Float32(
            [QF32 {
                data: [0.0; BLOCK],
            }; BLOCK_SIZE],
        )
    }
}

impl QData {
    pub fn qtype(&self) -> QTYPE {
        match self {
            QData::Binary(_) => QTYPE::Binary,
            QData::Int8(_) => QTYPE::Int8,
            QData::Float32(_) => QTYPE::Float32,
        }
    }

    pub fn dims(&self) -> usize {
        BLOCK_SIZE * BLOCK
    }
}
