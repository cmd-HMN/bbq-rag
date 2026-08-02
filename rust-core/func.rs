// Copyright 2024 cmd-HMN
//
// This file includes some or all code from the maxsim-cpu library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

mod function {
    use crate::cpu::vec256::simd;
    use crate::blas::*;

    pub fn pro_sgl_doc(
        _q : &[f32],
        _d : &[f32],
        _q_len: usize,
        _d_len: usize,
        _dim: usize
    ) -> f32 {
        unsafe {
        }
    }
}
