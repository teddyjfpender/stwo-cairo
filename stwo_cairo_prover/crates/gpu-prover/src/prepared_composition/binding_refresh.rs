//! Same-shape statement refresh for a persistent composition graph.

use core::ffi::c_void;

use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompositionBindingRefreshTelemetry {
    pub h2d_bytes: usize,
    pub h2d_copies: usize,
    pub sync_calls: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct CompositionBindingUpload {
    descriptor_offset_words: usize,
    words: Vec<u32>,
}

fn composition_binding_uploads(
    requirements: &CompositionWorkspaceRequirements,
    proof_bindings: &CompositionProofBindings,
) -> Result<Vec<CompositionBindingUpload>, PreparedCompositionError> {
    if proof_bindings.component_count() != requirements.components.len() {
        return Err(PreparedCompositionError::BaseParamBindingCount {
            expected: requirements.components.len(),
            actual: proof_bindings.component_count(),
        });
    }
    let expected_words = requirements
        .components
        .iter()
        .try_fold(0usize, |total, component| {
            total.checked_add(component.base_param_words)
        })
        .ok_or(PreparedCompositionError::SizeOverflow)?;
    if proof_bindings.base_param_word_count() != expected_words {
        return Err(PreparedCompositionError::BaseParamBindingTotalWords {
            expected: expected_words,
            actual: proof_bindings.base_param_word_count(),
        });
    }

    let mut uploads = Vec::new();
    for (component_index, (component, descriptor)) in requirements
        .components
        .iter()
        .zip(&requirements.component_descriptors)
        .enumerate()
    {
        let (binding_component, binding_instance, values) =
            proof_bindings.component(component_index).ok_or(
                PreparedCompositionError::BaseParamBindingIdentity(component_index),
            )?;
        if binding_component != component.component || binding_instance != component.instance {
            return Err(PreparedCompositionError::BaseParamBindingIdentity(
                component_index,
            ));
        }
        if values.len() != component.base_param_words {
            return Err(PreparedCompositionError::BaseParamBindingWords {
                component: component_index,
                expected: component.base_param_words,
                actual: values.len(),
            });
        }
        if !values.is_empty() {
            uploads.push(CompositionBindingUpload {
                descriptor_offset_words: descriptor.base_params,
                words: values.iter().map(|value| value.0).collect(),
            });
        }
    }
    Ok(uploads)
}

impl PreparedCompositionGraph<'_> {
    /// Refresh only the statement-varying base parameters of an already
    /// prepared same-shape graph. All pointers, layout words, denominators and
    /// AOT identities remain immutable; captured launches continue to read the
    /// same descriptor addresses.
    pub fn refresh_proof_bindings(
        &self,
        proof_bindings: &CompositionProofBindings,
    ) -> Result<CompositionBindingRefreshTelemetry, PreparedCompositionError> {
        let uploads = composition_binding_uploads(&self.requirements, proof_bindings)?;
        if uploads.is_empty() {
            return Ok(CompositionBindingRefreshTelemetry::default());
        }
        let h2d_bytes = uploads.iter().try_fold(0usize, |total, upload| {
            upload
                .words
                .len()
                .checked_mul(WORD_BYTES)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or(PreparedCompositionError::SizeOverflow)
        })?;
        let enqueue = (|| {
            for upload in &uploads {
                let destination = unsafe {
                    self.descriptors
                        .as_u32_ptr()
                        .add(upload.descriptor_offset_words)
                };
                unsafe {
                    self.arena.context().memcpy_h2d_async(
                        destination.cast::<c_void>(),
                        upload.words.as_ptr().cast::<c_void>(),
                        upload.words.len() * WORD_BYTES,
                    )?;
                }
            }
            Ok::<(), PreparedCompositionError>(())
        })();
        // Drain even after an enqueue failure: every host vector above must
        // remain alive until all earlier asynchronous copies have completed.
        let fence = self
            .arena
            .context()
            .sync()
            .map_err(PreparedCompositionError::from);
        enqueue.and(fence)?;
        Ok(CompositionBindingRefreshTelemetry {
            h2d_bytes,
            h2d_copies: uploads.len(),
            sync_calls: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::BaseField;
    use stwo::core::pcs::TreeSubspan;

    use super::*;

    fn plan(name: &'static str, values: &[u32]) -> CompositionPlan {
        CompositionPlan {
            max_kernel_instrs: 64,
            total_constraints: 1,
            max_evaluation_log_size: 5,
            components: vec![CompositionComponentPlan {
                component: name,
                instance: 0,
                trace_locations: vec![
                    TreeSubspan {
                        tree_index: 0,
                        col_start: 0,
                        col_end: 0,
                    },
                    TreeSubspan {
                        tree_index: 1,
                        col_start: 0,
                        col_end: 1,
                    },
                    TreeSubspan {
                        tree_index: 2,
                        col_start: 0,
                        col_end: 0,
                    },
                ],
                preprocessed_column_indices: vec![0],
                trace_log_size: 4,
                evaluation_log_size: 5,
                n_constraints: 1,
                random_coefficient_offset: 0,
                denominator_inverses: vec![BaseField::from(1); 2],
                base_param_values: values
                    .iter()
                    .copied()
                    .map(BaseField::from_u32_unchecked)
                    .collect(),
                ext_param_values: Vec::new(),
                ext_param_sources: Vec::new(),
                kernels: vec![CompositionKernelPart {
                    kernel_name: "kernel".to_owned(),
                    cache_key: 7,
                    semantic_hash: 9,
                    source: "extern \"C\" __global__ void kernel() {}".to_owned(),
                    rc_base: 0,
                }],
            }],
            wave_kernels: Vec::new(),
        }
    }

    fn requirements(plan: &CompositionPlan) -> CompositionWorkspaceRequirements {
        let tree = |slot, log_size| CompositionCoefficientSource {
            slot: ArenaSlotId(slot),
            log_size,
        };
        composition_workspace_requirements(
            plan,
            &CompositionTraceTopology {
                trees: vec![vec![tree(100, 4)], vec![tree(200, 4)], vec![]],
            },
        )
        .unwrap()
    }

    #[test]
    fn warm_refresh_is_exact_and_rejects_shape_drift() {
        let original = plan("component", &[7, 11]);
        let requirements = requirements(&original);
        let uploads = composition_binding_uploads(
            &requirements,
            &CompositionProofBindings::from_plan(&original),
        )
        .unwrap();
        assert_eq!(uploads.len(), 1);
        assert_eq!(
            uploads[0].descriptor_offset_words,
            requirements.component_descriptors[0].base_params
        );
        assert_eq!(uploads[0].words, vec![7, 11]);

        let changed = composition_binding_uploads(
            &requirements,
            &CompositionProofBindings::from_plan(&plan("component", &[13, 17])),
        )
        .unwrap();
        assert_eq!(
            changed[0].descriptor_offset_words,
            uploads[0].descriptor_offset_words
        );
        assert_eq!(changed[0].words, vec![13, 17]);

        assert_eq!(
            composition_binding_uploads(
                &requirements,
                &CompositionProofBindings::from_plan(&plan("different", &[7, 11])),
            ),
            Err(PreparedCompositionError::BaseParamBindingIdentity(0))
        );
        assert_eq!(
            composition_binding_uploads(
                &requirements,
                &CompositionProofBindings::from_plan(&plan("component", &[7])),
            ),
            Err(PreparedCompositionError::BaseParamBindingTotalWords {
                expected: 2,
                actual: 1,
            })
        );
    }
}
