use stwo_backend_cuda::{
    CompositionSplitLaunchMode, ModeAwareCommitWorkspaceRequirements, ModeAwareCommitWorkspaceSlots,
};

use super::*;

const SOURCE_COLUMNS: usize = 4;
const RETAINED_COLUMNS: usize = 8;

impl Compiler<'_> {
    pub(super) fn compile_split(&mut self) -> Result<(), CompositionAuthorityError> {
        let program = self
            .plan
            .composition()
            .output_plan
            .direct_program()
            .ok_or(CompositionAuthorityError::UnsupportedOutputMode)?;
        let schedule = program.schedule();
        if schedule.evaluation_log_size != 24
            || !program.admits_launch_mode(CompositionSplitLaunchMode::FusedFirstForward)
            || schedule.inverse_intervals != 3
            || schedule.final_inverse_first_stage != 19
            || schedule.final_inverse_stages != 6
            || schedule.first_forward_first_stage != 2
            || schedule.first_forward_stages != 5
            || schedule.remaining_forward_intervals != 2
        {
            return Err(CompositionAuthorityError::UnsupportedSplitMode(
                CompositionSplitLaunchMode::FusedFirstForward,
            ));
        }
        let rows = 1usize
            .checked_shl(schedule.evaluation_log_size)
            .ok_or(CompositionAuthorityError::SizeOverflow)?;
        let commitment = self.plan.commitment(CommitmentTreeId::Composition).ok_or(
            CompositionAuthorityError::MissingLogicalRole("composition commitment"),
        )?;
        let (requirements, slots) = match (&commitment.requirements, &commitment.slots) {
            (
                ModeAwareCommitWorkspaceRequirements::DomainProgressive(requirements),
                ModeAwareCommitWorkspaceSlots::DomainProgressive(slots),
            ) => (requirements, slots),
            _ => {
                return Err(CompositionAuthorityError::ShapeDrift(
                    "composition commit mode",
                ))
            }
        };
        let [batch] = requirements.leaves.plan.lde_batches.as_slice() else {
            return Err(CompositionAuthorityError::ShapeDrift(
                "composition commit batch",
            ));
        };
        let [batch_slots] = slots.leaves.batches.as_slice() else {
            return Err(CompositionAuthorityError::ShapeDrift(
                "composition commit batch slots",
            ));
        };
        if requirements.leaves.plan.columns.len() != RETAINED_COLUMNS
            || batch.columns != (0..RETAINED_COLUMNS).collect::<Vec<_>>()
            || batch.evaluation_log_size != schedule.evaluation_log_size
        {
            return Err(CompositionAuthorityError::ShapeDrift(
                "composition commit columns",
            ));
        }
        let [Some(output_bindings)] = commitment.evaluation_output_groups.as_slice() else {
            return Err(CompositionAuthorityError::ShapeDrift(
                "composition retained output group",
            ));
        };
        let outputs: [ArenaBinding; RETAINED_COLUMNS] = output_bindings
            .clone()
            .try_into()
            .map_err(|_| CompositionAuthorityError::ShapeDrift("eight retained outputs"))?;
        if outputs.iter().any(|binding| binding.len_words != rows) {
            return Err(CompositionAuthorityError::ShapeDrift(
                "retained output extent",
            ));
        }
        let source_pointers = self.binding_for_slot(
            BufferPurpose::CommitCoefficientPointers,
            batch_slots.coefficient_ptrs,
        )?;
        let retained_pointers =
            self.binding_for_slot(BufferPurpose::CommitOutputPointers, batch_slots.output_ptrs)?;
        self.insert_relocation(
            CompositionRelocationRole::SplitSourcePointers,
            source_pointers,
            0,
            SOURCE_COLUMNS * POINTER_WORDS,
            POINTER_WORDS,
        )?;
        self.insert_relocation(
            CompositionRelocationRole::SplitRetainedPointers,
            retained_pointers,
            0,
            RETAINED_COLUMNS * POINTER_WORDS,
            POINTER_WORDS,
        )?;
        for generation in 1..=3 {
            for (column, &binding) in outputs.iter().enumerate() {
                self.insert_binding(
                    CompositionValueRole::SplitRetained {
                        canonical_column: column as u8,
                        generation,
                    },
                    binding,
                    0,
                    rows,
                    1,
                )?;
            }
        }
        let max = *self
            .requirements
            .accumulators
            .last()
            .ok_or(CompositionAuthorityError::ShapeDrift("maximum accumulator"))?;
        if max.log_size != schedule.evaluation_log_size {
            return Err(CompositionAuthorityError::ShapeDrift(
                "split source accumulator",
            ));
        }
        let initial = self.generation(max.log_size)?;
        self.register_accumulator(max, initial + 1)?;
        self.register_accumulator(max, initial + 2)?;
        self.compile_split_inverse(max, initial, rows)?;
        self.compile_split_forward(max.log_size, rows)?;
        self.outputs = Some(std::array::from_fn(|column| {
            CompositionValueRole::SplitRetained {
                canonical_column: column as u8,
                generation: 3,
            }
        }));
        Ok(())
    }

    fn compile_split_inverse(
        &mut self,
        max: CompositionAccumulatorRequirements,
        initial: u8,
        rows: usize,
    ) -> Result<(), CompositionAuthorityError> {
        let mut children = Vec::with_capacity(3);
        children.push(self.inverse_in_place_child(
            "b2n_init_block_warp_batch<2>",
            max.log_size,
            initial,
            initial + 1,
            rows,
            CompositionLaunchGeometry {
                grid: [16_384, 1, 4],
                block: [32, 4, 1],
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
            vec![("start_stage", 1), ("stages", 10), ("coordinates", 4)],
        )?);
        children.push(self.inverse_in_place_child(
            "b2n_noinit_block_batch<4,false>",
            max.log_size,
            initial + 1,
            initial + 2,
            rows,
            CompositionLaunchGeometry {
                grid: [32, 64, 4],
                block: [32, 16, 1],
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
            vec![("start_stage", 11), ("end_stage", 18), ("coordinates", 4)],
        )?);
        let mut specs = Vec::new();
        for coordinate in 0..SOURCE_COLUMNS {
            specs.push(AccessSpec::read(
                CompositionValueRole::Accumulator {
                    log_size: max.log_size,
                    coordinate: coordinate as u8,
                    generation: initial + 2,
                },
                rows,
            ));
        }
        for column in 0..RETAINED_COLUMNS {
            specs.push(AccessSpec::write(
                CompositionValueRole::SplitRetained {
                    canonical_column: column as u8,
                    generation: 1,
                },
                rows,
            ));
        }
        specs.extend([
            AccessSpec::read(
                CompositionValueRole::InverseTwiddles,
                self.requirements.inverse_twiddle_words,
            ),
            AccessSpec::read(
                CompositionValueRole::ForwardTwiddles,
                self.requirements.forward_twiddle_words,
            ),
        ]);
        children.push(child(
            "composition_split_boundary_batch<3,true>",
            CompositionLaunchGeometry {
                grid: [8_192, 1, 4],
                block: [32, 8, 1],
                dynamic_shared_bytes: 0,
                cooperative: false,
            },
            vec![
                ("start_stage", 19),
                ("inverse_stages", 6),
                ("forward_first_stage", 2),
                ("forward_stages", 5),
            ],
            self.roles.effect(specs)?,
        )?);
        let source_values = (0..SOURCE_COLUMNS)
            .map(|coordinate| {
                global_index(
                    &children,
                    0,
                    CompositionValueRole::Accumulator {
                        log_size: max.log_size,
                        coordinate: coordinate as u8,
                        generation: initial,
                    },
                    false,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let retained_values = (0..RETAINED_COLUMNS)
            .map(|column| {
                global_index(
                    &children,
                    2,
                    CompositionValueRole::SplitRetained {
                        canonical_column: column as u8,
                        generation: 1,
                    },
                    true,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inverse = global_index(&children, 0, CompositionValueRole::InverseTwiddles, false)?;
        let forward = global_index(&children, 2, CompositionValueRole::ForwardTwiddles, false)?;
        self.operations.push(operation(
            self.source_identity,
            CompositionOperationKind::SplitInverseFusedFirstForward {
                evaluation_log_size: max.log_size,
            },
            CompositionAbi::SplitInverseFusedFirstForwardV1,
            vec![
                (
                    "source_values",
                    CompositionInvocationValue::PointerTable {
                        pointee_accesses: source_values,
                    },
                ),
                (
                    "retained_outputs",
                    CompositionInvocationValue::PointerTable {
                        pointee_accesses: retained_values,
                    },
                ),
                ("log_n", CompositionInvocationValue::U32(max.log_size)),
                (
                    "inverse_twiddles",
                    CompositionInvocationValue::Access(inverse),
                ),
                (
                    "inverse_twiddle_words",
                    CompositionInvocationValue::U32(to_u32(
                        self.requirements.inverse_twiddle_words,
                    )?),
                ),
                (
                    "forward_twiddles",
                    CompositionInvocationValue::Access(forward),
                ),
                (
                    "forward_twiddle_words",
                    CompositionInvocationValue::U32(to_u32(
                        self.requirements.forward_twiddle_words,
                    )?),
                ),
                (
                    "eval_domain_size",
                    CompositionInvocationValue::U32(1u32 << (max.log_size - 1)),
                ),
                ("stream", CompositionInvocationValue::ExecutionStream),
            ],
            Vec::new(),
            children,
            None,
        )?);
        Ok(())
    }

    fn inverse_in_place_child(
        &self,
        symbol: &'static str,
        log_size: u32,
        source_generation: u8,
        destination_generation: u8,
        rows: usize,
        launch: CompositionLaunchGeometry,
        parameters: Vec<(&'static str, u32)>,
    ) -> Result<CompositionChildLaunch, CompositionAuthorityError> {
        let mut specs = Vec::new();
        for coordinate in 0..SOURCE_COLUMNS {
            specs.push(AccessSpec::required_alias(
                CompositionValueRole::Accumulator {
                    log_size,
                    coordinate: coordinate as u8,
                    generation: source_generation,
                },
                CompositionValueRole::Accumulator {
                    log_size,
                    coordinate: coordinate as u8,
                    generation: destination_generation,
                },
                rows,
                InPlaceDiscipline::OrderedCompositeInPlace,
            ));
        }
        specs.push(AccessSpec::read(
            CompositionValueRole::InverseTwiddles,
            self.requirements.inverse_twiddle_words,
        ));
        child(symbol, launch, parameters, self.roles.effect(specs)?)
    }

    fn compile_split_forward(
        &mut self,
        log_size: u32,
        rows: usize,
    ) -> Result<(), CompositionAuthorityError> {
        let children = vec![
            self.forward_in_place_child(
                "n2b_nofinal_block_batch<4,4>",
                1,
                2,
                rows,
                CompositionLaunchGeometry {
                    grid: [32, 64, 8],
                    block: [32, 16, 1],
                    dynamic_shared_bytes: 0,
                    cooperative: false,
                },
                vec![("start_stage", 7), ("end_stage", 14), ("columns", 8)],
            )?,
            self.forward_in_place_child(
                "n2b_final_block_warp_batch<2,true>",
                2,
                3,
                rows,
                CompositionLaunchGeometry {
                    grid: [16_384, 1, 8],
                    block: [32, 4, 1],
                    dynamic_shared_bytes: 0,
                    cooperative: false,
                },
                vec![("start_stage", 15), ("stages", 10), ("columns", 8)],
            )?,
        ];
        let pointees = (0..RETAINED_COLUMNS)
            .map(|column| {
                global_index(
                    &children,
                    0,
                    CompositionValueRole::SplitRetained {
                        canonical_column: column as u8,
                        generation: 1,
                    },
                    false,
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let forward = global_index(&children, 0, CompositionValueRole::ForwardTwiddles, false)?;
        self.operations.push(operation(
            self.source_identity,
            CompositionOperationKind::SplitForwardAfterFirstInterval {
                evaluation_log_size: log_size,
            },
            CompositionAbi::SplitForwardAfterFirstIntervalV1,
            vec![
                (
                    "values",
                    CompositionInvocationValue::PointerTable {
                        pointee_accesses: pointees,
                    },
                ),
                ("log_n", CompositionInvocationValue::U32(log_size)),
                ("num_poly", CompositionInvocationValue::U32(8)),
                (
                    "forward_twiddles",
                    CompositionInvocationValue::Access(forward),
                ),
                (
                    "forward_twiddle_words",
                    CompositionInvocationValue::U32(to_u32(
                        self.requirements.forward_twiddle_words,
                    )?),
                ),
                (
                    "eval_domain_size",
                    CompositionInvocationValue::U32(1u32 << (log_size - 1)),
                ),
                ("stream", CompositionInvocationValue::ExecutionStream),
            ],
            Vec::new(),
            children,
            None,
        )?);
        Ok(())
    }

    fn forward_in_place_child(
        &self,
        symbol: &'static str,
        source_generation: u8,
        destination_generation: u8,
        rows: usize,
        launch: CompositionLaunchGeometry,
        parameters: Vec<(&'static str, u32)>,
    ) -> Result<CompositionChildLaunch, CompositionAuthorityError> {
        let mut specs = Vec::new();
        for column in 0..RETAINED_COLUMNS {
            specs.push(AccessSpec::required_alias(
                CompositionValueRole::SplitRetained {
                    canonical_column: column as u8,
                    generation: source_generation,
                },
                CompositionValueRole::SplitRetained {
                    canonical_column: column as u8,
                    generation: destination_generation,
                },
                rows,
                InPlaceDiscipline::OrderedCompositeInPlace,
            ));
        }
        specs.push(AccessSpec::read(
            CompositionValueRole::ForwardTwiddles,
            self.requirements.forward_twiddle_words,
        ));
        child(symbol, launch, parameters, self.roles.effect(specs)?)
    }
}

fn global_index(
    children: &[CompositionChildLaunch],
    child_index: usize,
    role: CompositionValueRole,
    destination: bool,
) -> Result<u32, CompositionAuthorityError> {
    let offset = children[..child_index]
        .iter()
        .try_fold(0usize, |total, child| {
            total.checked_add(child.effect.accesses.len())
        })
        .ok_or(CompositionAuthorityError::SizeOverflow)?;
    let local = access_index(&children[child_index].effect, role, destination)?;
    to_u32(
        offset
            .checked_add(local)
            .ok_or(CompositionAuthorityError::SizeOverflow)?,
    )
}
