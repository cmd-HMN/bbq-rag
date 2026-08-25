// Default will be Float32
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QTYPE {
    Binary,
    Int8,
    Float32
}

// default is f32 
impl Default for QTYPE {
    fn default() -> Self {
        QTYPE::Float32
    }
}

#[derive(Default, Debug)]                                                                                                                       
pub struct QData {
    qtype: QTYPE,
    data: Vec<QTYPE>,
    scale: Vec<QTYPE>
}

fn qf32_to_qi8(value: f32) -> QData {
     QData::default()
}
