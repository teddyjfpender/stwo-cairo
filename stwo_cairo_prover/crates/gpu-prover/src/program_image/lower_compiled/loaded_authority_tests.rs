use stwo_backend_cuda::ArenaSlotId;

use super::super::loaded_writer_binding::{
    validate_writer_binding, SliceBindingFields, WriterBindingFields, WriterBindingPlan,
};
use super::super::{producer_prefix, recorded_deduce_authority};
use super::*;

const MANIFEST: [u8; 32] = [4; 32];
const EXEC_CONTEXT: u64 = 11;
const STREAM: u64 = 12;
const DRIVER_CONTEXT: u64 = 13;
const MODULE: u64 = 14;
const FUNCTION: u64 = 15;

fn invocation(stateful: bool) -> RecordedWitnessInvocationShape {
    let mut deduce = recorded_deduce_authority::empty_for_test();
    if stateful {
        deduce.module_state =
            Some(recorded_deduce_authority::PedersenTableColumnsAndRowsV1::CANONICAL);
    }
    RecordedWitnessInvocationShape {
        program_identity: [1; 32],
        semantic_hash: 2,
        cache_key: 3,
        kernel_symbol: "recorded_witness".into(),
        abi_schema_identity: AotKernelAbiSchema::RecordedWitnessV1.identity(),
        deduce,
        launch: crate::compiled_proof::LaunchGeometry {
            grid: [1, 1, 1],
            block: [256, 1, 1],
            cluster: None,
            dynamic_shared_bytes: 0,
            cooperative: false,
        },
        source_arguments: (0..8)
            .map(|ordinal| SourceArgument::U32 {
                ordinal,
                value: if ordinal == 7 { 17 } else { 1 },
            })
            .collect(),
    }
}

fn function_publication(invocation: &RecordedWitnessInvocationShape) -> FunctionPublicationFields {
    FunctionPublicationFields {
        manifest_identity: MANIFEST,
        source_identity: invocation.deduce.source_identity,
        cubin_identity: [5; 32],
        program_identity: invocation.program_identity,
        abi_schema_identity: invocation.abi_schema_identity,
        authority_identity: [6; 32],
        kernel_symbol: invocation.kernel_symbol.clone(),
        semantic_hash: invocation.semantic_hash,
        cache_key: invocation.cache_key,
        target_sm: 86,
        device_ordinal: 0,
        driver_context_token: DRIVER_CONTEXT,
        module_token: MODULE,
        function_token: FUNCTION,
    }
}

fn installed_receipt(invocation: &RecordedWitnessInvocationShape) -> InstalledReceiptFields {
    InstalledReceiptFields {
        manifest_identity: MANIFEST,
        source_identity: invocation.deduce.source_identity,
        cubin_identity: [5; 32],
        program_identity: invocation.program_identity,
        abi_schema_identity: invocation.abi_schema_identity,
        authority_identity: [6; 32],
        kernel_symbol: invocation.kernel_symbol.clone(),
        semantic_hash: invocation.semantic_hash,
        cache_key: invocation.cache_key,
        target_sm: 86,
        abi_schema: AotKernelAbiSchema::RecordedWitnessV1,
        module_globals: expected_module_globals(invocation),
        ownership: InstalledAotFunctionOwnership::BorrowedPublished,
        launch_grid: invocation.launch.grid,
        launch_block: invocation.launch.block,
        dynamic_shared_bytes: invocation.launch.dynamic_shared_bytes,
        device_ordinal: 0,
        exec_context_token: EXEC_CONTEXT,
        driver_context_token: DRIVER_CONTEXT,
        module_token: MODULE,
        function_token: FUNCTION,
        stream_token: STREAM,
        function_publication: function_publication(invocation),
        pedersen_publication: None,
    }
}

fn validate_receipt(
    invocation: &RecordedWitnessInvocationShape,
    receipt: &InstalledReceiptFields,
    canonical: Option<&CanonicalPedersenFields>,
) -> Result<(), InvocationShapeError> {
    validate_installed_receipt(
        invocation,
        MANIFEST,
        0,
        8,
        6,
        EXEC_CONTEXT,
        STREAM,
        receipt,
        canonical,
    )
}

#[test]
fn writer_admission_rejects_detached_or_non_recorded_capabilities() {
    let invocation = invocation(false);
    assert_eq!(recorded_row_count(&invocation), Ok(17));
    let exact = WriterAdmissionFields {
        belongs_to_arena: true,
        kind: PreparedWriterKind::Recorded,
        row_count: 17,
        identity: WriterIdentityFields {
            label: "recorded_witness".into(),
            kernel_name: invocation.kernel_symbol.clone(),
            semantic_hash: invocation.semantic_hash,
            cache_key: invocation.cache_key,
            aot_manifest_identity: MANIFEST,
            mode: PreparedWitnessMode::RequireEmbeddedAot,
        },
        has_installed_receipt: true,
    };
    validate_writer_admission(&invocation, "recorded_witness", 17, &exact).unwrap();

    let mutations: [fn(&mut WriterAdmissionFields); 10] = [
        |fields| fields.belongs_to_arena = false,
        |fields| fields.kind = PreparedWriterKind::BlakeGFused,
        |fields| fields.kind = PreparedWriterKind::BlakeGDirect,
        |fields| fields.row_count += 1,
        |fields| fields.identity.label.push('x'),
        |fields| fields.identity.kernel_name.push('x'),
        |fields| fields.identity.semantic_hash ^= 1,
        |fields| fields.identity.cache_key ^= 1,
        |fields| fields.identity.aot_manifest_identity = [0; 32],
        |fields| fields.identity.mode = PreparedWitnessMode::PreResolved,
    ];
    for mutate in mutations {
        let mut changed = exact.clone();
        mutate(&mut changed);
        assert_eq!(
            validate_writer_admission(&invocation, "recorded_witness", 17, &changed),
            Err(InvocationShapeError::LoadedAotAuthorityMismatch)
        );
    }
    let mut missing = exact;
    missing.has_installed_receipt = false;
    assert_eq!(
        validate_writer_admission(&invocation, "recorded_witness", 17, &missing),
        Err(InvocationShapeError::MissingLoadedAotAuthority)
    );

    let mut malformed = invocation.clone();
    malformed.source_arguments.pop();
    assert_eq!(
        recorded_row_count(&malformed),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    let SourceArgument::U32 { ordinal, value } = &mut malformed.source_arguments[6] else {
        unreachable!()
    };
    *ordinal = 7;
    *value = 0;
    malformed.source_arguments.push(SourceArgument::U32 {
        ordinal: 7,
        value: 0,
    });
    assert_eq!(
        recorded_row_count(&malformed),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
}

fn slice(seed: u32) -> SliceBindingFields {
    SliceBindingFields {
        slot: ArenaSlotId(seed),
        offset_words: seed as usize * 100,
        words: seed as usize + 10,
        pointer_token: u64::from(seed) * 1_000 + 1,
    }
}

fn writer_binding() -> WriterBindingFields {
    WriterBindingFields {
        input_columns: vec![slice(1), slice(2)],
        output_columns: vec![slice(3)],
        multiplicity_columns: Vec::new(),
        lookup_words: slice(4),
        sub_words: slice(5),
        descriptors: (6..11).map(slice).collect(),
        multiplicity_dummy: Some(slice(11)),
    }
}

#[test]
fn writer_binding_rejects_every_slice_identity_and_order_mutation() {
    let exact = writer_binding();
    validate_writer_binding(&exact, &exact).unwrap();
    let mut mutations = Vec::new();
    let slice_mutations: [fn(&mut SliceBindingFields); 4] = [
        |field: &mut SliceBindingFields| field.slot.0 += 1,
        |field: &mut SliceBindingFields| field.offset_words += 1,
        |field: &mut SliceBindingFields| field.words += 1,
        |field: &mut SliceBindingFields| field.pointer_token += 1,
    ];
    for mutate in slice_mutations {
        let mut changed = exact.clone();
        mutate(&mut changed.input_columns[0]);
        mutations.push(changed);
    }
    let mut changed = exact.clone();
    changed.input_columns.swap(0, 1);
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.output_columns[0] = slice(20);
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.multiplicity_columns.push(slice(21));
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.lookup_words = slice(22);
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.sub_words = slice(23);
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.descriptors.swap(0, 1);
    mutations.push(changed);
    let mut changed = exact.clone();
    changed.multiplicity_dummy = None;
    mutations.push(changed);
    for changed in mutations {
        assert_eq!(
            validate_writer_binding(&exact, &changed),
            Err(InvocationShapeError::LoadedAotAuthorityMismatch)
        );
    }
}

#[test]
fn canonical_planned_writer_bindings_equal_source_argument_bindings() {
    let executable = super::super::tests::generated_sn2_replacement();
    let arena = executable.arena();
    let catalog = BaseProducerCatalog::compile(arena).unwrap();
    let authority = executable.replacement_base_producers().unwrap();
    let mut recorded = 0;
    for producer in &authority.producers {
        let producer_prefix::SemanticBaseProducer::Recorded(producer) = producer else {
            continue;
        };
        recorded += 1;
        let planned = arena
            .witness()
            .components
            .iter()
            .find(|planned| {
                planned.component == producer.producer.component
                    && Some(planned.part) == producer.producer.part
            })
            .unwrap();
        let planned_binding = WriterBindingPlan::from_planned(planned).unwrap();
        assert_eq!(
            planned_binding,
            WriterBindingPlan::from_invocation(&producer.source, &catalog).unwrap()
        );

        let mut changed = producer.source.clone();
        let SourceArgument::PointerTable { entries, .. } = &mut changed.source_arguments[1] else {
            unreachable!()
        };
        let target = entries
            .iter_mut()
            .find(|entry| entry.target.elements.end != 0)
            .unwrap();
        target.target.elements.end -= 1;
        assert!(WriterBindingPlan::from_invocation(&changed, &catalog).is_err());

        let mut changed = producer.source.clone();
        let SourceArgument::ScalarArray { entries, .. } = &mut changed.source_arguments[2] else {
            unreachable!()
        };
        entries[1].value += 1;
        assert!(WriterBindingPlan::from_invocation(&changed, &catalog).is_err());

        let mut changed = producer.source.clone();
        let SourceArgument::PointerTable { ordinal, .. } = &mut changed.source_arguments[0] else {
            unreachable!()
        };
        *ordinal = 7;
        assert!(WriterBindingPlan::from_invocation(&changed, &catalog).is_err());

        let mut changed = producer.source.clone();
        let SourceArgument::U32 { value, .. } = &mut changed.source_arguments[7] else {
            unreachable!()
        };
        *value += 1;
        assert_ne!(
            WriterBindingPlan::from_invocation(&changed, &catalog).unwrap(),
            planned_binding
        );

        let mut changed = (*planned).clone();
        changed.slots.input_columns[0] = changed.slots.output_columns[0];
        assert_ne!(
            WriterBindingPlan::from_planned(&changed).unwrap(),
            planned_binding
        );
    }
    assert!(recorded > 0);
}

#[test]
fn installed_receipt_rejects_top_level_launch_context_and_publication_mutations() {
    let invocation = invocation(false);
    let exact = installed_receipt(&invocation);
    validate_receipt(&invocation, &exact, None).unwrap();

    let mutations: [fn(&mut InstalledReceiptFields); 30] = [
        |r| r.manifest_identity[0] ^= 1,
        |r| r.source_identity[0] ^= 1,
        |r| r.cubin_identity = [0; 32],
        |r| r.program_identity[0] ^= 1,
        |r| r.abi_schema_identity[0] ^= 1,
        |r| r.authority_identity = [0; 32],
        |r| r.kernel_symbol.push('x'),
        |r| r.semantic_hash ^= 1,
        |r| r.cache_key ^= 1,
        |r| r.target_sm += 1,
        |r| r.abi_schema = AotKernelAbiSchema::OrdinaryConstraintV1,
        |r| r.module_globals = AotKernelModuleGlobals::WitnessPedersenV1,
        |r| r.launch_grid[0] += 1,
        |r| r.launch_grid[1] += 1,
        |r| r.launch_grid[2] += 1,
        |r| r.launch_block[0] += 1,
        |r| r.launch_block[1] += 1,
        |r| r.launch_block[2] += 1,
        |r| r.dynamic_shared_bytes += 1,
        |r| r.device_ordinal += 1,
        |r| r.exec_context_token += 1,
        |r| r.driver_context_token = 0,
        |r| r.module_token = 0,
        |r| r.function_token = 0,
        |r| r.stream_token += 1,
        |r| r.function_publication.manifest_identity[0] ^= 1,
        |r| r.function_publication.source_identity[0] ^= 1,
        |r| r.function_publication.cubin_identity[0] ^= 1,
        |r| r.function_publication.program_identity[0] ^= 1,
        |r| r.function_publication.abi_schema_identity[0] ^= 1,
    ];
    for mutate in mutations {
        let mut changed = exact.clone();
        mutate(&mut changed);
        assert_eq!(
            validate_receipt(&invocation, &changed, None),
            Err(InvocationShapeError::LoadedAotAuthorityMismatch)
        );
    }
    let publication_mutations: [fn(&mut FunctionPublicationFields); 9] = [
        |f| f.authority_identity[0] ^= 1,
        |f| f.kernel_symbol.push('x'),
        |f| f.semantic_hash ^= 1,
        |f| f.cache_key ^= 1,
        |f| f.target_sm += 1,
        |f| f.device_ordinal += 1,
        |f| f.driver_context_token += 1,
        |f| f.module_token += 1,
        |f| f.function_token += 1,
    ];
    for mutate in publication_mutations {
        let mut changed = exact.clone();
        mutate(&mut changed.function_publication);
        assert_eq!(
            validate_receipt(&invocation, &changed, None),
            Err(InvocationShapeError::LoadedAotAuthorityMismatch)
        );
    }

    let mut changed = invocation.clone();
    changed.launch.cluster = Some([1, 1, 1]);
    assert_eq!(
        validate_receipt(&changed, &exact, None),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    let mut changed = invocation.clone();
    changed.launch.cooperative = true;
    assert_eq!(
        validate_receipt(&changed, &exact, None),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    let mut changed = invocation.clone();
    let mut changed_receipt = exact.clone();
    changed.launch.grid[0] = 0;
    changed_receipt.launch_grid[0] = 0;
    assert_eq!(
        validate_receipt(&changed, &changed_receipt, None),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    let mut changed = invocation.clone();
    let mut changed_receipt = exact.clone();
    changed.launch.block[0] = 0;
    changed_receipt.launch_block[0] = 0;
    assert_eq!(
        validate_receipt(&changed, &changed_receipt, None),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    assert_eq!(
        validate_installed_receipt(
            &invocation,
            [0; 32],
            0,
            8,
            6,
            EXEC_CONTEXT,
            STREAM,
            &exact,
            None,
        ),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
    assert_eq!(
        validate_installed_receipt(
            &invocation,
            MANIFEST,
            0,
            8,
            7,
            EXEC_CONTEXT,
            STREAM,
            &exact,
            None,
        ),
        Err(InvocationShapeError::LoadedAotAuthorityMismatch)
    );
}

fn canonical_pedersen() -> CanonicalPedersenFields {
    let state = recorded_deduce_authority::PedersenTableColumnsAndRowsV1::CANONICAL;
    CanonicalPedersenFields {
        content_digest: PedersenTableContentDigest::new([21; 32]),
        source_rows: state.resource.registered_source_rows as usize,
        padded_rows: state.resource.registered_padded_rows as usize,
        registration_generation: PEDERSEN_TABLE_REGISTRATION_GENERATION,
        column_pointers: (1..=state.resource.columns).map(u64::from).collect(),
    }
}

fn pedersen_publication(
    invocation: &RecordedWitnessInvocationShape,
    canonical: &CanonicalPedersenFields,
) -> PedersenPublicationFields {
    let state = invocation.deduce.module_state.unwrap();
    PedersenPublicationFields {
        manifest_identity: MANIFEST,
        source_identity: invocation.deduce.source_identity,
        cubin_identity: [5; 32],
        program_identity: invocation.program_identity,
        abi_schema_identity: invocation.abi_schema_identity,
        authority_identity: [6; 32],
        kernel_symbol: invocation.kernel_symbol.clone(),
        semantic_hash: invocation.semantic_hash,
        cache_key: invocation.cache_key,
        target_sm: 86,
        table_content_digest: canonical.content_digest,
        table_source_rows: canonical.source_rows,
        table_padded_rows: canonical.padded_rows,
        table_registration_generation: canonical.registration_generation,
        device_ordinal: 0,
        module_token: MODULE,
        function_token: FUNCTION,
        context_token: DRIVER_CONTEXT,
        columns_symbol_token: 31,
        columns_symbol_bytes: state.column_pointers.symbol_bytes,
        rows_symbol_token: 32,
        rows_symbol_bytes: state.row_count.symbol_bytes,
        completion_event_token: 33,
        column_pointers: canonical.column_pointers.clone(),
    }
}

#[test]
fn nested_pedersen_receipt_rejects_every_authority_and_table_mutation() {
    let stateful = invocation(true);
    let canonical = canonical_pedersen();
    let mut exact = installed_receipt(&stateful);
    exact.pedersen_publication = Some(pedersen_publication(&stateful, &canonical));
    validate_receipt(&stateful, &exact, Some(&canonical)).unwrap();

    let mut missing = exact.clone();
    missing.pedersen_publication = None;
    assert_eq!(
        validate_receipt(&stateful, &missing, Some(&canonical)),
        Err(InvocationShapeError::MissingLoadedModuleStateAuthority)
    );
    assert_eq!(
        validate_receipt(&stateful, &exact, None),
        Err(InvocationShapeError::MissingLoadedModuleStateAuthority)
    );

    let mutations: [fn(&mut PedersenPublicationFields); 24] = [
        |p| p.manifest_identity[0] ^= 1,
        |p| p.source_identity[0] ^= 1,
        |p| p.cubin_identity[0] ^= 1,
        |p| p.program_identity[0] ^= 1,
        |p| p.abi_schema_identity[0] ^= 1,
        |p| p.authority_identity[0] ^= 1,
        |p| p.kernel_symbol.push('x'),
        |p| p.semantic_hash ^= 1,
        |p| p.cache_key ^= 1,
        |p| p.target_sm += 1,
        |p| p.table_content_digest = PedersenTableContentDigest::new([99; 32]),
        |p| p.table_source_rows += 1,
        |p| p.table_padded_rows += 1,
        |p| p.table_registration_generation += 1,
        |p| p.device_ordinal += 1,
        |p| p.module_token += 1,
        |p| p.function_token += 1,
        |p| p.context_token += 1,
        |p| p.columns_symbol_token = 0,
        |p| p.columns_symbol_bytes += 1,
        |p| p.rows_symbol_token = 0,
        |p| p.rows_symbol_bytes += 1,
        |p| p.completion_event_token = 0,
        |p| p.column_pointers[0] += 1,
    ];
    for mutate in mutations {
        let mut changed = exact.clone();
        mutate(changed.pedersen_publication.as_mut().unwrap());
        assert_eq!(
            validate_receipt(&stateful, &changed, Some(&canonical)),
            Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
        );
    }
    let mut changed = exact.clone();
    let publication = changed.pedersen_publication.as_mut().unwrap();
    publication.rows_symbol_token = publication.columns_symbol_token;
    assert_eq!(
        validate_receipt(&stateful, &changed, Some(&canonical)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );

    let mut changed = canonical.clone();
    changed.content_digest = PedersenTableContentDigest::new([88; 32]);
    assert_eq!(
        validate_receipt(&stateful, &exact, Some(&changed)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );
    let mut changed = canonical.clone();
    changed.column_pointers.swap(0, 1);
    assert_eq!(
        validate_receipt(&stateful, &exact, Some(&changed)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );
    let mut changed = canonical.clone();
    changed.source_rows += 1;
    assert_eq!(
        validate_receipt(&stateful, &exact, Some(&changed)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );
    let mut changed = canonical.clone();
    changed.padded_rows += 1;
    assert_eq!(
        validate_receipt(&stateful, &exact, Some(&changed)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );
    let mut changed = canonical.clone();
    changed.registration_generation += 1;
    assert_eq!(
        validate_receipt(&stateful, &exact, Some(&changed)),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );

    let stateless = invocation(false);
    let mut unexpected = installed_receipt(&stateless);
    unexpected.pedersen_publication = exact.pedersen_publication;
    assert_eq!(
        validate_receipt(&stateless, &unexpected, None),
        Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch)
    );
}
