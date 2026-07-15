use stwo::core::fft::butterfly;
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::poly::utils::domain_line_twiddles_from_tree;
use stwo::prover::backend::cpu::CpuBackend;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};

use super::*;

fn layer(values: &mut [BaseField], i: usize, h: usize, t: BaseField) {
    for l in 0..(1 << i) {
        let left = (h << (i + 1)) + l;
        let right = left + (1 << i);
        let (mut a, mut b) = (values[left], values[right]);
        butterfly(&mut a, &mut b, t);
        (values[left], values[right]) = (a, b);
    }
}

fn evaluate_after_duplicate(coefficients: &[BaseField]) -> Vec<BaseField> {
    let coefficient_log = coefficients.len().ilog2();
    let evaluation_log = coefficient_log + 1;
    let domain = CanonicCoset::new(evaluation_log).circle_domain();
    let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
    let line_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);
    let mut values = coefficients
        .iter()
        .chain(coefficients)
        .copied()
        .collect::<Vec<_>>();

    // The highest line layer was replaced by the duplicate write above.
    for (line_index, line) in line_twiddles.iter().enumerate().rev().skip(1) {
        for (h, &twiddle) in line.iter().enumerate() {
            layer(&mut values, line_index + 1, h, twiddle);
        }
    }
    for (h, pair) in line_twiddles[0].chunks_exact(2).enumerate() {
        let x = pair[0];
        let y = pair[1];
        for (offset, twiddle) in [y, -y, -x, x].into_iter().enumerate() {
            layer(&mut values, 0, h * 4 + offset, twiddle);
        }
    }
    values
}

#[test]
fn duplicate_first_stage_is_byte_exact_against_cpu_oracle() {
    for coefficient_log in 3..=12 {
        for seed in 1..=8u32 {
            let mut state = seed.wrapping_mul(0x9e37_79b9);
            let coefficients = (0..1u32 << coefficient_log)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    BaseField::from(state)
                })
                .collect::<Vec<_>>();
            let evaluation_log = coefficient_log + 1;
            let domain = CanonicCoset::new(evaluation_log).circle_domain();
            let twiddles = CpuBackend::precompute_twiddles(domain.half_coset);
            let expected = CpuBackend::evaluate(
                &CircleCoefficients::new(coefficients.clone()),
                domain,
                &twiddles,
            );
            assert_eq!(evaluate_after_duplicate(&coefficients), expected.values);
        }
    }
}

#[test]
fn duplicate_first_partitions_cover_all_remaining_stages() {
    for log in 2..=30 {
        assert_eq!(
            duplicate_first_intervals(log).unwrap().iter().sum::<u32>(),
            log - 1
        );
    }
}

#[test]
fn only_logs_20_and_28_remove_an_additional_full_pass() {
    let reduced = (13..=30)
        .filter(|&log| {
            duplicate_first_intervals(log).unwrap().len()
                < current_n2b_intervals(log).unwrap().len()
        })
        .collect::<Vec<_>>();
    assert_eq!(reduced, [20, 28]);
}

#[test]
fn butterfly_with_zero_is_twiddle_independent_duplication() {
    let value = BaseField::from(123_456_789);
    for twiddle in [0, 1, 7, 1_000_003].map(BaseField::from) {
        let (mut left, mut right) = (value, BaseField::from(0));
        butterfly(&mut left, &mut right, twiddle);
        assert_eq!((left, right), (value, value));
    }
}
