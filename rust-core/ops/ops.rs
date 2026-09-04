// /! All maxsim variants used in this project

pub mod function {
    use crate::cpu::{
        dm_f32, dm_i8
    };

    use crate::quantization::{QParmas, QTYPE};
    use rayon::prelude::{IntoParallelIterator, ParallelIterator};

    #[derive(Debug, Clone)]
    pub enum DocLayout<'a> {
        Flat { d_len: &'a [usize] },
        Single { doc_tokens: usize },
        Batch { docs: usize, tokens: usize },
    }

    fn process_single_doc(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        d_len: usize,
        q_scale_ptr: usize,
        d_scale_ptr: usize,
        dim: usize,
        dtype: QTYPE,
    ) -> f32 {
        match dtype {
            QTYPE::Float32 => {
                let q_slice =
                    unsafe { std::slice::from_raw_parts(q_ptr as *const f32, q_len * dim) };

                let d_slice =
                    unsafe { std::slice::from_raw_parts(d_ptr as *const f32, d_len * dim) };

                dm_f32(q_slice, d_slice, q_len, d_len, dim)
            }
            QTYPE::Int8 => {
                let num_blocks = dim / QParmas::BLOCK;

                let q_slice: &[i8] =
                    unsafe { std::slice::from_raw_parts(q_ptr as *const i8, q_len * dim) };

                let d_slice: &[i8] =
                    unsafe { std::slice::from_raw_parts(d_ptr as *const i8, d_len * dim) };

                static DUMMY_UNIT_SCALE: [f32; 1024] = [1.0f32; 1024];

                let q_scale: &[f32] = if q_scale_ptr != 0 {
                    unsafe {
                        std::slice::from_raw_parts(q_scale_ptr as *const f32, q_len * num_blocks)
                    }
                } else {
                    &DUMMY_UNIT_SCALE[..(q_len * num_blocks).min(1024)]
                };

                let d_scale: &[f32] = if d_scale_ptr != 0 {
                    unsafe {
                        std::slice::from_raw_parts(d_scale_ptr as *const f32, d_len * num_blocks)
                    }
                } else {
                    &DUMMY_UNIT_SCALE[..(d_len * num_blocks).min(1024)]
                };

                dm_i8(q_slice, d_slice, q_scale, d_scale, q_len, d_len, dim)
            }
            _ => {
                panic!("Unsupported type, only Float32 and Int8 supported");
            }
        }
    }

    fn execute_jobs<I, T, F>(items: I, jobs: i32, op: F) -> Vec<T>
    where
        I: IntoIterator + IntoParallelIterator<Item = <I as IntoIterator>::Item> + Send,
        <I as IntoIterator>::Item: Send,
        T: Send,
        F: Fn(<I as IntoIterator>::Item) -> T + Sync + Send,
    {
        if jobs == -1 {
            items.into_par_iter().map(op).collect()
        } else if jobs > 1 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(jobs as usize)
                .build()
                .expect("Failed to build custom rayon threadpool");
            pool.install(|| items.into_par_iter().map(op).collect())
        } else {
            items.into_iter().map(op).collect()
        }
    }

    #[inline(always)]
    fn process_single(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        d_len: usize,
        q_scale_ptr: usize,
        d_scale_ptr: usize,
        dim: usize,
        dtype: QTYPE,
    ) -> Vec<f32> {
        if q_len == 0 || d_len == 0 {
            return Vec::new();
        }

        vec![process_single_doc(
            q_ptr,
            d_ptr,
            q_len,
            d_len,
            q_scale_ptr,
            d_scale_ptr,
            dim,
            dtype,
        )]
    }

    #[inline(always)]
    fn process_batch(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        docs: usize,
        tokens: usize,
        jobs: i32,
        q_scale_ptr: usize,
        d_scale_ptr: usize,
        dim: usize,
        dtype: QTYPE,
    ) -> Vec<f32> {
        if q_len == 0 || docs == 0 || tokens == 0 {
            return Vec::new();
        }

        let blocks = dim / QParmas::BLOCK;
        let elem_size = match dtype {
            QTYPE::Float32 => std::mem::size_of::<f32>(),
            QTYPE::Int8 => std::mem::size_of::<i8>(),
            _ => panic!("Unsupported type, only Float32 and Int8 supported"),
        };

        let scale_elem_size = std::mem::size_of::<f32>();

        let doc_stride_bytes = tokens * dim * elem_size;
        let scale_stride_bytes = tokens * blocks * scale_elem_size;

        let run = |idx: usize| {
            let cur_d_ptr = d_ptr + idx * doc_stride_bytes;
            let cur_d_scale_ptr = if d_scale_ptr != 0 {
                d_scale_ptr + idx * scale_stride_bytes
            } else {
                0
            };

            process_single_doc(
                q_ptr,
                cur_d_ptr,
                q_len,
                tokens,
                q_scale_ptr,
                cur_d_scale_ptr,
                dim,
                dtype,
            )
        };

        execute_jobs(0..docs, jobs, run)
    }

    #[inline(always)]
    fn process_flat(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        d_len: &[usize],
        jobs: i32,
        q_scale_ptr: usize,
        d_scale_ptr: usize,
        dim: usize,
        dtype: QTYPE,
    ) -> Vec<f32> {
        let n_docs = d_len.len();
        if q_len == 0 || n_docs == 0 {
            return Vec::new();
        }

        let blocks = dim / QParmas::BLOCK;
        let elem_size = match dtype {
            QTYPE::Float32 => std::mem::size_of::<f32>(),
            QTYPE::Int8 => std::mem::size_of::<i8>(),
            _ => panic!("Unsupported type, only Float32 and Int8 supported"),
        };

        let scale_elem_size = std::mem::size_of::<f32>();

        let mut offset = Vec::with_capacity(n_docs);
        let mut curr_token = 0usize;
        for &len in d_len {
            offset.push(curr_token);
            curr_token += len;
        }

        let run = |idx: usize| {
            let doc_tokens = d_len[idx];
            if doc_tokens == 0 {
                return 0.0f32;
            }

            let cur_d_ptr = d_ptr + offset[idx] * dim * elem_size;
            let cur_d_scale_ptr = if d_scale_ptr != 0 {
                d_scale_ptr + offset[idx] * blocks * scale_elem_size
            } else {
                0
            };

            process_single_doc(
                q_ptr,
                cur_d_ptr,
                q_len,
                doc_tokens,
                q_scale_ptr,
                cur_d_scale_ptr,
                dim,
                dtype,
            )
        };

        execute_jobs(0..n_docs, jobs, run)
    }

    pub unsafe fn maxsim(
        q_ptr: usize,
        d_ptr: usize,
        q_len: usize,
        q_scale_ptr: usize,
        d_scale_ptr: usize,
        dim: usize,
        layout: DocLayout<'_>,
        dtype: QTYPE,
        jobs: i32,
    ) -> Vec<f32> {
        match layout {
            DocLayout::Single { doc_tokens } => {
                process_single(
                    q_ptr,
                    d_ptr,
                    q_len,
                    doc_tokens,
                    q_scale_ptr,
                    d_scale_ptr,
                    dim,
                    dtype,
                )
            }
            DocLayout::Batch { docs, tokens } => {
                process_batch(
                    q_ptr,
                    d_ptr,
                    q_len,
                    docs,
                    tokens,
                    jobs,
                    q_scale_ptr,
                    d_scale_ptr,
                    dim,
                    dtype,
                )
            }
            DocLayout::Flat { d_len } => {
                process_flat(
                    q_ptr,
                    d_ptr,
                    q_len,
                    d_len,
                    jobs,
                    q_scale_ptr,
                    d_scale_ptr,
                    dim,
                    dtype,
                )
            }
        }
    }
}
