/// Contains the sgemm function
/// # Arguments
/// * `side` - The side of the matrix
/// * `uplo` - The upper or lower triangle of the matrix
/// * `m` - The number of rows in the matrix
/// * `n` - The number of columns in the matrix
/// * `k` - The number of columns in the other matrix
/// * `alpha` - The scalar multiplier
/// * `a` - The first matrix
/// * `lda` - The leading dimension of the first matrix
/// * `b` - The second matrix
/// * `ldb` - The leading dimension of the second matrix
/// * `beta` - The scalar multiplier
/// * `c` - The result matrix
/// * `ldc` - The leading dimension of the result matrix

/// Reference
/// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-1/gemm.html

#[cfg(target_arch = "x86_64")]
pub mod mkl_blas {
    use libc::c_char;

    extern crate intel_mkl_src;

    mod ffi {
        use libc::c_char;
        unsafe extern "C" {
            pub fn sgemm_(
                side: *const c_char,
                uplo: *const c_char,
                m: *const i32,
                n: *const i32,
                k: *const i32,
                alpha: *const f32,
                a: *const f32,
                lda: *const i32,
                b: *const f32,
                ldb: *const i32,
                beta: *const f32,
                c: *mut f32,
                ldc: *const i32,
            );
        }
    }

    #[inline]
    pub unsafe fn sgemm(
        side: u8,
        uplo: u8,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
        unsafe {
            ffi::sgemm_(
                &(side as c_char),
                &(uplo as c_char),
                &m,
                &n,
                &k,
                &alpha,
                a.as_ptr(),
                &lda,
                b.as_ptr(),
                &ldb,
                &beta,
                c.as_mut_ptr(),
                &ldc,
            )
        }
    }
}

mod custom {
    use std::arch::x86_64::*;

    /// Kernel Size
    const MR: usize = 6;
    const NR: usize = 16;

    // -------------------------------------
    // Sgemm funciton
    // C = alpha * A * B + beta * C
    // To handle matrix based multiplication
    // inspired by gemm from intel mkl
    //
    // Uses col major appraoch as in BLAS
    // # Observation
    // Good for matrix mutiplication
    // Bad for vector dot product
    //
    // # Inspired
    // - https://cs.stanford.edu/people/shadjis/blas.html
    // - https://salykova.github.io/matmul-cpu
    // -------------------------------------

    /// scale_c function
    /// # Arguments
    /// * `c` - The result matrix
    /// * `beta` - The scalar multiplier
    /// * `n` - The number of columns in the matrix
    ///
    /// # Formula
    /// C = beta * C -> Use in sgemm
    ///
    /// # Returns
    /// None
    #[inline(always)]
    unsafe fn scale_c(beta: f32, m: usize, n: usize, c: &mut [f32], ldc: usize) {
        if beta == 0.0 {
            for j in 0..n {
                for i in 0..m {
                    c[i + j * ldc] = 0.0;
                }
            }
        } else if beta != 1.0 {
            for j in 0..n {
                for i in 0..m {
                    c[i + j * ldc] *= beta;
                }
            }
        }
    }

    /// packed_col_a_mr_blocked function
    /// # Arguments
    /// * `a` - The first matrix
    /// * `n` - The number of rows in the matrix
    /// * `m` - The number of columns in the matrix
    /// * `lda` - The leading dimension of the first matrix
    /// * `kernel` - The kernel size
    /// * `result` - The result matrix
    ///
    /// # Returns
    /// None
    #[inline(always)]
    unsafe fn packed_col_a_mr_blocked(
        _a: &[f32],
        _n: usize,
        _m: usize,
        _lda: usize,
        _kernel: usize, //MR
        _result: &mut [f32],
    ) {
        for _i in (0.._n).step_by(_kernel) {
            let mr = (_i + _kernel).min(_n) - _i;
            for _j in 0.._m {
                let base_idx = _i * _m + _j * _kernel;
                for _k in 0..mr {
                    _result[base_idx + _k] = _a[_j * _lda + _i + _k];
                }
                // Add the padding
                for _k in mr.._kernel {
                    _result[base_idx + _k] = 0.0;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn packed_col_a_transpose_mr_blocked(
        _a: &[f32],
        _n: usize,
        _m: usize,
        _lda: usize,
        _kernel: usize, //MR
        _result: &mut [f32],
    ) {
        for _i in (0.._n).step_by(_kernel) {
            let mr = (_i + _kernel).min(_n) - _i;
            for _j in 0.._m {
                let base_idx = _i * _m + _j * _kernel;
                for _k in 0..mr {
                    _result[base_idx + _k] = _a[_j + _lda * (_i + _k)];
                }
                // Add the padding
                for _k in mr.._kernel {
                    _result[base_idx + _k] = 0.0;
                }
            }
        }
    }

    /// packed_col_b_nr_blocked function
    /// # Arguments
    /// * `b` - The second matrix
    /// * `k` - The number of columns in the other matrix
    /// * `m` - The number of rows in the matrix
    /// * `ldb` - The leading dimension of the second matrix
    /// * `kernel` - The kernel size
    /// * `result` - The result matrix
    ///
    /// # Returns
    /// None
    #[inline(always)]
    unsafe fn packed_col_b_nr_blocked(
        _b: &[f32],
        _k: usize,
        _m: usize,
        _ldb: usize,
        _kernel: usize, //NR
        _result: &mut [f32],
    ) {
        for _j in (0.._m).step_by(_kernel) {
            let nr = (_j + _kernel).min(_m) - _j;
            for _i in 0.._k {
                let base_idx = _j * _k + _i * _kernel;
                for _ii in 0..nr {
                    _result[base_idx + _ii] = _b[(_j + _ii) * _ldb + _i];
                }
                // Add the padding
                for _ii in nr.._kernel {
                    _result[base_idx + _ii] = 0.0;
                }
            }
        }
    }

    /// packed_col_b_transpose_nr_blocked function
    /// # Arguments
    /// * `b` - The second matrix
    /// * `k` - The number of columns in the other matrix
    /// * `m` - The number of rows in the matrix
    /// * `ldb` - The leading dimension of the second matrix
    /// * `kernel` - The kernel size
    /// * `result` - The result matrix
    ///
    /// # Returns
    /// None
    #[inline(always)]
    unsafe fn packed_col_b_transpose_nr_blocked(
        _b: &[f32],
        _k: usize,
        _m: usize,
        _ldb: usize,
        _kernel: usize,
        _result: &mut [f32],
    ) {
        for _j in (0.._m).step_by(_kernel) {
            let nr = (_j + _kernel).min(_m) - _j;
            for _i in 0.._k {
                let base_idx = _j * _k + _i * _kernel;
                for _ii in 0..nr {
                    // Column-major source: (j + jj) * ldb + i
                    _result[base_idx + _ii] = _b[(_j + _ii) + _ldb * _i];
                }
                // Add the padding
                for _ii in nr.._kernel {
                    _result[base_idx + _ii] = 0.0;
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    /// mk_6x16 function
    /// # Arguments
    /// * `m` - The number of rows in the matrix
    /// * `a` - The first matrix
    /// * `b` - The second matrix
    /// * `c` - The result matrix
    ///
    /// # Returns
    /// None
    unsafe fn mk_6x16(_m: usize, _a: &[f32], _b: &[f32], _c: &mut [f32]) {
        unsafe {
            // setting based on 6x16 (registers)
            let mut _c0 = _mm256_setzero_ps();
            let mut _c1 = _mm256_setzero_ps();

            let mut _c2 = _mm256_setzero_ps();
            let mut _c3 = _mm256_setzero_ps();

            let mut _c4 = _mm256_setzero_ps();
            let mut _c5 = _mm256_setzero_ps();

            let mut _c6 = _mm256_setzero_ps();
            let mut _c7 = _mm256_setzero_ps();

            let mut _c8 = _mm256_setzero_ps();
            let mut _c9 = _mm256_setzero_ps();

            let mut _c10 = _mm256_setzero_ps();
            let mut _c11 = _mm256_setzero_ps();

            for p in 0.._m {
                let _b0 = _mm256_loadu_ps(_b.as_ptr().add(p * NR));
                let _b1 = _mm256_loadu_ps(_b.as_ptr().add(p * NR + 8));

                let _a0 = _mm256_broadcast_ss(_a.get_unchecked(p * MR));
                _c0 = _mm256_fmadd_ps(_a0, _b0, _c0);
                _c1 = _mm256_fmadd_ps(_a0, _b1, _c1);

                let _a1 = _mm256_broadcast_ss(_a.get_unchecked(p * MR + 1));
                _c2 = _mm256_fmadd_ps(_a1, _b0, _c2);
                _c3 = _mm256_fmadd_ps(_a1, _b1, _c3);

                let _a2 = _mm256_broadcast_ss(_a.get_unchecked(p * MR + 2));
                _c4 = _mm256_fmadd_ps(_a2, _b0, _c4);
                _c5 = _mm256_fmadd_ps(_a2, _b1, _c5);

                let _a3 = _mm256_broadcast_ss(_a.get_unchecked(p * MR + 3));
                _c6 = _mm256_fmadd_ps(_a3, _b0, _c6);
                _c7 = _mm256_fmadd_ps(_a3, _b1, _c7);

                let _a4 = _mm256_broadcast_ss(_a.get_unchecked(p * MR + 4));
                _c8 = _mm256_fmadd_ps(_a4, _b0, _c8);
                _c9 = _mm256_fmadd_ps(_a4, _b1, _c9);

                let _a5 = _mm256_broadcast_ss(_a.get_unchecked(p * MR + 5));
                _c10 = _mm256_fmadd_ps(_a5, _b0, _c10);
                _c11 = _mm256_fmadd_ps(_a5, _b1, _c11);
            }

            // now store in _c
            _mm256_storeu_ps(_c.as_mut_ptr().add(0), _c0);
            _mm256_storeu_ps(_c.as_mut_ptr().add(8), _c1);
            _mm256_storeu_ps(_c.as_mut_ptr().add(16), _c2);
            _mm256_storeu_ps(_c.as_mut_ptr().add(24), _c3);
            _mm256_storeu_ps(_c.as_mut_ptr().add(32), _c4);
            _mm256_storeu_ps(_c.as_mut_ptr().add(40), _c5);
            _mm256_storeu_ps(_c.as_mut_ptr().add(48), _c6);
            _mm256_storeu_ps(_c.as_mut_ptr().add(56), _c7);
            _mm256_storeu_ps(_c.as_mut_ptr().add(64), _c8);
            _mm256_storeu_ps(_c.as_mut_ptr().add(72), _c9);
            _mm256_storeu_ps(_c.as_mut_ptr().add(80), _c10);
            _mm256_storeu_ps(_c.as_mut_ptr().add(88), _c11);
        }
    }

    #[inline(always)]
    unsafe fn update_c(
        _mr: usize,
        _nr: usize,
        _alpha: f32,
        _beta: f32,
        _c: &mut [f32],
        _ldc: usize,
        _c_pack: &[f32],
    ) {
        if _beta == 0.0 {
            for _j in 0.._nr {
                for _i in 0.._mr {
                    _c[_j * _ldc + _i] = _alpha * _c_pack[_i * NR + _j];
                }
            }
        } else if _beta == 1.0 {
            for _j in 0.._nr {
                for _i in 0.._mr {
                    _c[_j * _ldc + _i] += _alpha * _c_pack[_i * NR + _j];
                }
            }
        } else {
            for _j in 0.._nr {
                for _i in 0.._mr {
                    _c[_j * _ldc + _i] =
                        _alpha * _c_pack[_i * NR + _j] + _beta * _c[_j * _ldc + _i];
                }
            }
        }
    }

    // As this is only for col pali so only supprt nn and nt format for sgemm
    #[inline(always)]
    unsafe fn sgemm_nn_block<const MC: usize, const NC: usize, const KC: usize>(
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &[f32],
        _lda: u32,
        _b: &[f32],
        _ldb: u32,
        _beta: f32,
        _c: &mut [f32],
        _ldc: u32,
    ) {
        let mut _a_pack = vec![0.0f32; MC * KC];
        let mut _b_pack = vec![0.0f32; NC * KC];

        let mut _c_pack = vec![0.0f32; MR * NR];

        // using 5 loop arch
        for _jnc in (0.._n).step_by(NC) {
            let _nc = (_jnc + NC).min(_n) - _jnc;

            let mut _betam = _beta;

            for _pkc in (0.._k).step_by(KC) {
                let _kc = (_pkc + KC).min(_k) - _pkc;
                // b
                unsafe {
                    packed_col_b_nr_blocked(
                        &_b[_pkc + _jnc * (_ldb as usize)..],
                        _kc as usize,
                        _nc as usize,
                        _ldb as usize,
                        NR,
                        &mut _b_pack,
                    )
                }

                for _imc in (0.._m).step_by(MC) {
                    let _mc = (_imc + MC).min(_m) - _imc;

                    // now a
                    unsafe {
                        packed_col_a_mr_blocked(
                            &_a[_imc + _pkc * (_lda as usize)..],
                            _mc as usize,
                            _kc as usize,
                            _lda as usize,
                            MR,
                            &mut _a_pack,
                        );
                    }

                    for _jnr in (0.._nc).step_by(NR) {
                        let _nr = (_jnr + NR).min(_nc) - _jnr;

                        for _imr in (0.._mc).step_by(MR) {
                            let _mr = (_imr + MR).min(_mc) - _imr;

                            unsafe {
                                mk_6x16(
                                    _kc,
                                    &_a_pack[_imr * _kc..],
                                    &_b_pack[_jnr * _kc..],
                                    &mut _c_pack,
                                );
                                update_c(
                                    _mr,
                                    _nr,
                                    _alpha,
                                    _betam,
                                    &mut _c[(_jnc + _jnr) * (_ldc as usize) + (_imc + _imr)..],
                                    _ldc as usize,
                                    &mut _c_pack,
                                );
                            }
                        }
                    }
                }
                _betam = 1.0;
            }
        }
    }

    #[inline(always)]
    unsafe fn sgemm_nt_block<const MC: usize, const NC: usize, const KC: usize>(
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &[f32],
        _lda: u32,
        _b: &[f32],
        _ldb: u32,
        _beta: f32,
        _c: &mut [f32],
        _ldc: u32,
    ) {
        let mut _a_pack = vec![0.0f32; MC * KC];
        let mut _b_pack = vec![0.0f32; NC * KC];

        let mut _c_pack = vec![0.0f32; MR * NR];

        // using 5 loop arch
        for _jnc in (0.._n).step_by(NC) {
            let _nc = (_jnc + NC).min(_n) - _jnc;

            let mut _betam = _beta;

            for _pkc in (0.._k).step_by(KC) {
                let _kc = (_pkc + KC).min(_k) - _pkc;
                // b
                unsafe {
                    packed_col_b_transpose_nr_blocked(
                        &_b[_jnc + _pkc * (_ldb as usize)..],
                        _kc as usize,
                        _nc as usize,
                        _ldb as usize,
                        NR,
                        &mut _b_pack,
                    )
                }

                for _imc in (0.._m).step_by(MC) {
                    let _mc = (_imc + MC).min(_m) - _imc;

                    // now a
                    unsafe {
                        packed_col_a_mr_blocked(
                            &_a[_imc + _pkc * (_lda as usize)..],
                            _mc as usize,
                            _kc as usize,
                            _lda as usize,
                            MR,
                            &mut _a_pack,
                        );
                    }

                    for _jnr in (0.._nc).step_by(NR) {
                        let _nr = (_jnr + NR).min(_nc) - _jnr;

                        for _imr in (0.._mc).step_by(MR) {
                            let _mr = (_imr + MR).min(_mc) - _imr;

                            unsafe {
                                mk_6x16(
                                    _kc,
                                    &_a_pack[_imr * _kc..],
                                    &_b_pack[_jnr * _kc..],
                                    &mut _c_pack,
                                );
                                update_c(
                                    _mr,
                                    _nr,
                                    _alpha,
                                    _betam,
                                    &mut _c[(_jnc + _jnr) * (_ldc as usize) + (_imc + _imr)..],
                                    _ldc as usize,
                                    &mut _c_pack,
                                );
                            }
                        }
                    }
                }
                _betam = 1.0;
            }
        }
    }

    // As this is only for col pali so only supprt nn and nt format for sgemm
    #[inline(always)]
    unsafe fn sgemm_tn_block<const MC: usize, const NC: usize, const KC: usize>(
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &[f32],
        _lda: u32,
        _b: &[f32],
        _ldb: u32,
        _beta: f32,
        _c: &mut [f32],
        _ldc: u32,
    ) {
        let mut _a_pack = vec![0.0f32; MC * KC];
        let mut _b_pack = vec![0.0f32; NC * KC];

        let mut _c_pack = vec![0.0f32; MR * NR];

        // using 5 loop arch
        for _jnc in (0.._n).step_by(NC) {
            let _nc = (_jnc + NC).min(_n) - _jnc;

            let mut _betam = _beta;

            for _pkc in (0.._k).step_by(KC) {
                let _kc = (_pkc + KC).min(_k) - _pkc;
                // b
                unsafe {
                    packed_col_b_nr_blocked(
                        &_b[_pkc + _jnc * (_ldb as usize)..],
                        _kc as usize,
                        _nc as usize,
                        _ldb as usize,
                        NR,
                        &mut _b_pack,
                    )
                }

                for _imc in (0.._m).step_by(MC) {
                    let _mc = (_imc + MC).min(_m) - _imc;

                    // now a
                    unsafe {
                        packed_col_a_transpose_mr_blocked(
                            &_a[_pkc + _imc * (_lda as usize)..],
                            _mc as usize,
                            _kc as usize,
                            _lda as usize,
                            MR,
                            &mut _a_pack,
                        );
                    }

                    for _jnr in (0.._nc).step_by(NR) {
                        let _nr = (_jnr + NR).min(_nc) - _jnr;

                        for _imr in (0.._mc).step_by(MR) {
                            let _mr = (_imr + MR).min(_mc) - _imr;

                            unsafe {
                                mk_6x16(
                                    _kc,
                                    &_a_pack[_imr * _kc..],
                                    &_b_pack[_jnr * _kc..],
                                    &mut _c_pack,
                                );
                                update_c(
                                    _mr,
                                    _nr,
                                    _alpha,
                                    _betam,
                                    &mut _c[(_jnc + _jnr) * (_ldc as usize) + (_imc + _imr)..],
                                    _ldc as usize,
                                    &mut _c_pack,
                                );
                            }
                        }
                    }
                }
                _betam = 1.0;
            }
        }
    }

    /// Sgemm function
    /// C = alpha * A * B + beta * C
    ///
    /// # Arguments
    /// * `side` - The side of the matrix
    /// * `uplo` - The upper or lower triangle of the matrix
    /// * `m` - The number of rows in the matrix
    /// * `n` - The number of columns in the matrix
    /// * `k` - The number of columns in the other matrix
    /// * `alpha` - The scalar multiplier
    /// * `a` - The first matrix
    /// * `lda` - The leading dimension of the first matrix
    /// * `b` - The second matrix
    /// * `ldb` - The leading dimension of the second matrix
    /// * `beta` - The scalar multiplier
    /// * `c` - The result matrix
    /// * `ldc` - The leading dimension of the result matrix
    ///
    /// # Returns
    /// None
    #[inline(always)]
    pub fn _f32_mm(
        _transa: usize,
        _transb: usize,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &[f32],
        _lda: u32,
        _b: &[f32],
        _ldb: u32,
        _beta: f32,
        _c: &mut [f32],
        _ldc: u32,
    ) {
        let _nota = _transa as u8 == b'N' || _transa as u8 == b'n';
        let _notb = _transb as u8 == b'N' || _transb as u8 == b'n';

        assert!(
            _a.len()
                >= if _nota {
                    _lda as usize * _k
                } else {
                    _lda as usize * _m
                }
        );
        assert!(
            _b.len()
                >= if _notb {
                    _ldb as usize * _n
                } else {
                    _ldb as usize * _k
                }
        );
        assert!(_c.len() >= _ldc as usize * _n);
        if _m == 0 || _n == 0 || ((_alpha == 0.0 || _k == 0) && _beta == 1.0) {
            return;
        }

        if _alpha == 0.0 || _k == 0 {
            unsafe {
                scale_c(_beta, _m, _n, _c, _ldc as usize);
                return;
            }
        }
        unsafe {
            match (_nota, _notb) {
                (true, true) => sgemm_nn_block::<72, 256, 256>(
                    _m, _n, _k, _alpha, _a, _lda, _b, _ldb, _beta, _c, _ldc,
                ),
                (true, false) => sgemm_nt_block::<72, 256, 256>(
                    _m, _n, _k, _alpha, _a, _lda, _b, _ldb, _beta, _c, _ldc,
                ),
                (false, true) => sgemm_tn_block::<72, 256, 256>(
                    _m, _n, _k, _alpha, _a, _lda, _b, _ldb, _beta, _c, _ldc,
                ),
                _ => panic!("Unsupported case"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_sgemm_test(transa: u8, transb: u8, m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
        let lda = if transa == b'N' || transa == b'n' {
            m
        } else {
            k
        };
        let ldb = if transb == b'N' || transb == b'n' {
            k
        } else {
            n
        };
        let ldc = m;

        let a: Vec<f32> = (0..m * k).map(|x| x as f32 % 7.0).collect();
        let b: Vec<f32> = (0..k * n).map(|x| x as f32 % 7.0).collect();

        let mut c_custom: Vec<f32> = (0..m * n).map(|x| x as f32 % 3.0).collect();
        let mut c_mkl = c_custom.clone();

        custom::_f32_mm(
            transa as usize,
            transb as usize,
            m,
            n,
            k,
            alpha,
            &a,
            lda as u32,
            &b,
            ldb as u32,
            beta,
            &mut c_custom,
            ldc as u32,
        );

        unsafe {
            mkl_blas::sgemm(
                transa, transb, m as i32, n as i32, k as i32, alpha, &a, lda as i32, &b,
                ldb as i32, beta, &mut c_mkl, ldc as i32,
            );
        }

        let epsilon = 1e-3;

        for i in 0..(m * n) {
            let diff = (c_custom[i] - c_mkl[i]).abs();
            assert!(
                diff < epsilon,
                "Mismatch at index {}! Custom: {}, MKL: {} (TransA: {}, TransB: {}, M: {}, N: {}, K: {})",
                i,
                c_custom[i],
                c_mkl[i],
                transa as char,
                transb as char,
                m,
                n,
                k
            );
        }
    }

    #[test]
    fn test_01_standard_square() {
        run_sgemm_test(b'N', b'N', 256, 256, 256, 1.0, 0.0);
    }

    #[test]
    fn test_02_small_micro_kernel_only() {
        run_sgemm_test(b'N', b'N', 16, 16, 16, 1.0, 0.0);
    }

    #[test]
    fn test_03_odd_dimensions_fringe() {
        run_sgemm_test(b'N', b'N', 17, 19, 23, 1.0, 0.0);
    }

    #[test]
    fn test_04_tall_and_skinny() {
        run_sgemm_test(b'N', b'N', 512, 16, 64, 1.0, 0.0);
    }

    #[test]
    fn test_05_short_and_wide() {
        run_sgemm_test(b'N', b'N', 16, 512, 64, 1.0, 0.0);
    }

    #[test]
    fn test_06_deep_k_accumulation() {
        run_sgemm_test(b'N', b'N', 64, 64, 1024, 1.0, 0.0);
    }

    #[test]
    fn test_07_beta_accumulation() {
        run_sgemm_test(b'N', b'N', 128, 128, 128, 1.0, 1.0);
    }

    #[test]
    fn test_08_alpha_beta_fractional_scaling() {
        run_sgemm_test(b'N', b'N', 64, 64, 64, 0.5, 0.5);
    }

    #[test]
    fn test_09_zero_alpha() {
        run_sgemm_test(b'N', b'N', 128, 128, 128, 0.0, 1.0);
    }

    #[test]
    fn test_10_prime_numbers_large() {
        run_sgemm_test(b'N', b'N', 73, 257, 263, 1.0, 0.0);
    }

    #[test]
    fn test_11_nt_standard_square() {
        run_sgemm_test(b'n', b't', 128, 128, 128, 1.0, 0.0);
    }

    #[test]
    fn test_12_nt_odd_dimensions() {
        run_sgemm_test(b'n', b't', 17, 19, 23, 1.0, 0.0);
    }

    #[test]
    fn test_13_nt_tall_and_skinny() {
        run_sgemm_test(b'n', b't', 256, 16, 64, 1.0, 0.0);
    }

    #[test]
    fn test_14_tn_standard_square() {
        run_sgemm_test(b't', b'n', 128, 128, 128, 1.0, 0.0);
    }

    #[test]
    fn test_15_tn_odd_dimensions() {
        run_sgemm_test(b't', b'n', 17, 19, 23, 1.0, 0.0);
    }

    #[test]
    fn test_16_tn_tall_and_skinny() {
        run_sgemm_test(b't', b'n', 256, 16, 64, 1.0, 0.0);
    }

    #[test]
    fn test_17_nt_multiblock_large() {
        run_sgemm_test(b'n', b't', 64, 512, 320, 1.0, 0.0);
    }
}
