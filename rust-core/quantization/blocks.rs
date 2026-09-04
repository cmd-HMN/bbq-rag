pub struct QParmas;

impl QParmas {
    pub const BASE_DIMS: usize = 128;
    pub const BLOCK: usize = 32;
    pub const BLOCK_SIZE: usize = QParmas::BASE_DIMS / QParmas::BLOCK;
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QF32 {
    pub data: [f32; QParmas::BLOCK],
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QI8 {
    pub scale: f32,
    pub data: [i8; QParmas::BLOCK],
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QBin {
    pub data: [u8; QParmas::BLOCK],
}

// Default will be Float32
// THis of for block level quantization type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QTYPE {
    Binary,
    Int8,
    #[default]
    Float32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QData {
    Binary([QBin; QParmas::BLOCK_SIZE]),
    Int8([QI8; QParmas::BLOCK_SIZE]),
    Float32([QF32; QParmas::BLOCK_SIZE]),
}

impl Default for QData {
    fn default() -> Self {
        QData::Float32(
            [QF32 {
                data: [0.0; QParmas::BLOCK],
            }; QParmas::BLOCK_SIZE],
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
        QParmas::BLOCK_SIZE * QParmas::BLOCK
    }
}

pub struct QBlock {
    pub qtype: QTYPE,
    pub qdata: QData,
}

impl Default for QBlock {
    fn default() -> Self {
        QBlock {
            qtype: QTYPE::default(),
            qdata: QData::default(),
        }
    }
}

impl QBlock {
    pub fn new(qtype: QTYPE, qdata: QData) -> Self {
        QBlock { qtype, qdata }
    }

    pub fn get_dim(&self) -> usize {
        QParmas::BASE_DIMS
    }

    // so that we can convert QData to ([T; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE])
    pub fn to_array<T>(&self) -> Result<([T; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE]), &'static str>
    where
        ([T; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE]): TryFrom<QData, Error = &'static str>,
    {
        self.qdata.try_into()
    }

    // so that we can convert QData to [T; QParmas::BASE_DIMS]
    pub fn to_values<T>(&self) -> Result<[T; QParmas::BASE_DIMS], &'static str>
    where
        [T; QParmas::BASE_DIMS]: TryFrom<QData, Error = &'static str>,
    {
        self.qdata.try_into()
    }
}

// for type i8 with scale
impl TryFrom<QData> for ([i8; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE]) {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Int8(blocks) => {
                let mut out_data = [0i8; QParmas::BASE_DIMS];
                let mut out_scale = [0f32; QParmas::BLOCK_SIZE];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out_data[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                    out_scale[i] = blk.scale;
                }

                Ok((out_data, out_scale))
            }
            _ => Err("You are initializing with wrong datatype, its not Int8"),
        }
    }
}

// for type f32 with scale
impl TryFrom<QData> for ([f32; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE]) {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Float32(blocks) => {
                let mut out_data = [0f32; QParmas::BASE_DIMS];
                let out_scale = [1.0f32; QParmas::BLOCK_SIZE];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out_data[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                }

                Ok((out_data, out_scale))
            }
            _ => Err("You are initializing with wrong datatype, its not Float32"),
        }
    }
}

// for type u8 with scale
impl TryFrom<QData> for ([u8; QParmas::BASE_DIMS], [f32; QParmas::BLOCK_SIZE]) {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Binary(blocks) => {
                let mut out_data = [0u8; QParmas::BASE_DIMS];
                let out_scale = [1.0f32; QParmas::BLOCK_SIZE];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out_data[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                }

                Ok((out_data, out_scale))
            }
            _ => Err("You are initializing with wrong datatype, its not Binary"),
        }
    }
}

// for type f32
impl TryFrom<QData> for [f32; QParmas::BASE_DIMS] {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Float32(blocks) => {
                let mut out = [0f32; QParmas::BASE_DIMS];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                }

                Ok(out)
            }
            _ => Err("You are initializing with wrong datatype, its not Float32"),
        }
    }
}

// for type i8
impl TryFrom<QData> for [i8; QParmas::BASE_DIMS] {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Int8(blocks) => {
                let mut out = [0i8; QParmas::BASE_DIMS];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                }

                Ok(out)
            }
            _ => Err("You are initializing with wrong datatype, its not Int8"),
        }
    }
}

impl TryFrom<QData> for [u8; QParmas::BASE_DIMS] {
    type Error = &'static str;

    fn try_from(qdata: QData) -> Result<Self, Self::Error> {
        match qdata {
            QData::Binary(blocks) => {
                let mut out = [0u8; QParmas::BASE_DIMS];

                for (i, blk) in blocks.iter().enumerate() {
                    let start = i * QParmas::BLOCK;
                    out[start..start + QParmas::BLOCK].copy_from_slice(&blk.data);
                }

                Ok(out)
            }
            _ => Err("You are initializing with wrong datatype, its not Binary"),
        }
    }
}
