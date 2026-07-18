//! Runtime admission for address-free registered fixed-source requirements.
//!
//! The compiled authority names the canonical recipe and geometry. This module
//! separately requires the process registration returned by Cairo's canonical
//! byte-checking path, retains its byte digest and live addresses, and resolves
//! one relocation per module initializer. Per-module symbol publication,
//! context, and completion remain attested by each loaded recorded-writer
//! receipt; those process facts never enter the compiled identity.

use std::collections::BTreeSet;

use stwo_backend_cuda::aot;
use stwo_backend_cuda::pedersen_table::{
    registered_borrowed_pedersen_table, PedersenTableContentDigest, RegisteredPedersenColumn,
    RegisteredPedersenTable, PEDERSEN_TABLE_REGISTRATION_GENERATION,
};

use super::super::compiled_base_prefix::module_globals::{
    registered_pedersen_source, resolve as resolve_module_globals,
};
use super::super::producer_prefix::{BaseProducerAuthority, SemanticBaseProducer};
use super::super::recorded_deduce_authority::PedersenTableColumnsAndRowsV1;
use super::super::resolved_recorded_build_authority::ResolvedRecordedBuildAuthority;
use super::super::InvocationShapeError;
use crate::compiled_proof::{
    ByteRange, ModuleGlobalInitializer, ModuleGlobalInitializerAtom, ModuleGlobalInitializerId,
    ModuleIdentity, RegisteredFixedSourceAuthority, RegisteredFixedSourceRead,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegisteredColumnSnapshot {
    index: usize,
    address: u64,
    elements: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegisteredSourceSnapshot {
    byte_digest: PedersenTableContentDigest,
    source_rows: usize,
    padded_rows: usize,
    element_bytes: usize,
    generation: u64,
    columns: Vec<RegisteredColumnSnapshot>,
}

impl RegisteredSourceSnapshot {
    fn from_pedersen(table: RegisteredPedersenTable) -> Result<Self, InvocationShapeError> {
        table
            .validate_exact_registration_geometry(
                table.content_digest(),
                table.source_n_rows(),
                table.n_rows(),
            )
            .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
        Ok(Self {
            byte_digest: table.content_digest(),
            source_rows: table.source_n_rows(),
            padded_rows: table.n_rows(),
            element_bytes: core::mem::size_of::<u32>(),
            generation: table.registration_generation(),
            columns: table
                .columns()
                .into_iter()
                .map(|column| {
                    Ok(RegisteredColumnSnapshot {
                        index: column.index(),
                        address: u64::try_from(column.as_u32_ptr() as usize)
                            .map_err(|_| InvocationShapeError::SizeOverflow)?,
                        elements: column.len_words(),
                    })
                })
                .collect::<Result<_, InvocationShapeError>>()?,
        })
    }
}

/// Two independent views of the one admitted process source.
///
/// `registered` is the already-published singleton. `canonical` is returned by
/// re-entering Cairo's byte-digest-checked canonical registration path. Exact
/// equality prevents geometry-only admission of a foreign ready singleton.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedRegisteredFixedSource {
    registered: RegisteredSourceSnapshot,
    canonical: RegisteredSourceSnapshot,
}

impl PreparedRegisteredFixedSource {
    fn canonical_pedersen18() -> Result<Self, InvocationShapeError> {
        let registered = registered_borrowed_pedersen_table()
            .ok_or(InvocationShapeError::MissingLoadedModuleStateAuthority)?;
        let canonical =
            stwo_cairo_prover::witness::jit_prove_backend::try_ensure_device_pedersen_table()
                .map_err(|_| InvocationShapeError::MissingLoadedModuleStateAuthority)?;
        Ok(Self {
            registered: RegisteredSourceSnapshot::from_pedersen(registered)?,
            canonical: RegisteredSourceSnapshot::from_pedersen(canonical)?,
        })
    }

    /// Bind one direct fixed-table read to the exact checked registration and
    /// the live column borrowed by the prepared graph.
    pub(in crate::program_image::lower_compiled) fn validate_read_column(
        &self,
        read: &RegisteredFixedSourceRead,
        column: RegisteredPedersenColumn,
    ) -> Result<(), InvocationShapeError> {
        let column_address = u64::try_from(column.as_u32_ptr() as usize)
            .map_err(|_| InvocationShapeError::SizeOverflow)?;
        self.validate_read_column_facts(read, column.index(), column_address, column.len_words())
    }

    fn validate_read_column_facts(
        &self,
        read: &RegisteredFixedSourceRead,
        column_index: usize,
        column_address: u64,
        column_elements: usize,
    ) -> Result<(), InvocationShapeError> {
        let canonical = registered_pedersen_source(PedersenTableColumnsAndRowsV1::CANONICAL)
            .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
        bind_source(canonical.clone(), self)?;
        if read.source() != &canonical
            || read.column() != column_index
            || read.elements().start != 0
            || read.elements().end != canonical.padded_rows()
            || self.registered != self.canonical
            || self.canonical.generation != PEDERSEN_TABLE_REGISTRATION_GENERATION
        {
            return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
        }
        let snapshot = self
            .canonical
            .columns
            .get(read.column())
            .filter(|snapshot| {
                snapshot.index == read.column()
                    && snapshot.elements == read.elements().end
                    && snapshot.address != 0
                    && snapshot.address == column_address
                    && column_elements == read.elements().len()
            })
            .ok_or(InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
        if snapshot.address % self.canonical.element_bytes as u64 != 0 {
            return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedRegisteredFixedSource {
    authority: RegisteredFixedSourceAuthority,
    byte_digest: PedersenTableContentDigest,
    generation: u64,
    column_addresses: Box<[u64]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedRegisteredFixedSourceRelocation {
    initializer: ModuleGlobalInitializerId,
    module: ModuleIdentity,
    symbol: Box<[u8]>,
    destination: ByteRange,
    source_identity: [u8; 32],
    column_addresses: Box<[u64]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadedRegisteredModuleGlobals {
    sources: Vec<LoadedRegisteredFixedSource>,
    relocations: Vec<LoadedRegisteredFixedSourceRelocation>,
}

pub(super) fn prepare_for_base(
    authority: &BaseProducerAuthority,
) -> Result<Vec<PreparedRegisteredFixedSource>, InvocationShapeError> {
    match canonical_required_sources(required_base_sources(authority)?)?.len() {
        0 => Ok(Vec::new()),
        1 => Ok(vec![PreparedRegisteredFixedSource::canonical_pedersen18()?]),
        _ => Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch),
    }
}

pub(super) fn bind_for_base(
    authority: &BaseProducerAuthority,
    prepared: &[PreparedRegisteredFixedSource],
    sm_major: u32,
    sm_minor: u32,
) -> Result<LoadedRegisteredModuleGlobals, InvocationShapeError> {
    let initializers = canonical_base_initializers(authority, sm_major, sm_minor)?;
    let expected_relocations = initializers
        .iter()
        .flat_map(|initializer| initializer.atoms())
        .filter(|atom| {
            matches!(
                atom,
                ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses { .. }
            )
        })
        .count();
    let loaded = bind_module_initializers(&initializers, prepared)?;
    if loaded.relocations.len() != expected_relocations {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    Ok(loaded)
}

/// Resolve every registered-source atom while deduplicating only its immutable
/// process table. Module relocations remain distinct.
fn bind_module_initializers(
    initializers: &[ModuleGlobalInitializer],
    prepared: &[PreparedRegisteredFixedSource],
) -> Result<LoadedRegisteredModuleGlobals, InvocationShapeError> {
    let mut initializer_ids = BTreeSet::new();
    let mut required = Vec::new();
    for initializer in initializers {
        if !initializer_ids.insert(initializer.id())
            || !initializer
                .has_valid_identity()
                .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?
        {
            return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
        }
        for atom in initializer.atoms() {
            if let ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination,
                source,
            } = atom
            {
                let state = PedersenTableColumnsAndRowsV1::CANONICAL;
                let symbol_bytes = usize::try_from(state.column_pointers.symbol_bytes)
                    .map_err(|_| InvocationShapeError::SizeOverflow)?;
                if initializer.symbol() != state.column_pointers.symbol.as_bytes()
                    || initializer.bytes() != symbol_bytes
                    || initializer.alignment()
                        != usize::try_from(state.column_pointers.alignment_bytes)
                            .map_err(|_| InvocationShapeError::SizeOverflow)?
                    || *destination
                        != ByteRange::new(0, symbol_bytes)
                            .ok_or(InvocationShapeError::LoadedModuleStateAuthorityMismatch)?
                {
                    return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
                }
                required.push(source.clone());
            }
        }
    }
    let sources = bind_sources(required, prepared)?;
    let mut relocations = Vec::new();
    for initializer in initializers {
        for atom in initializer.atoms() {
            let ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination,
                source,
            } = atom
            else {
                continue;
            };
            let loaded = sources
                .iter()
                .find(|loaded| loaded.authority == *source)
                .ok_or(InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
            relocations.push(LoadedRegisteredFixedSourceRelocation {
                initializer: initializer.id(),
                module: initializer.module().clone(),
                symbol: initializer.symbol().into(),
                destination: *destination,
                source_identity: *source.identity(),
                column_addresses: loaded.column_addresses.clone(),
            });
        }
    }
    Ok(LoadedRegisteredModuleGlobals {
        sources,
        relocations,
    })
}

fn canonical_base_initializers(
    authority: &BaseProducerAuthority,
    sm_major: u32,
    sm_minor: u32,
) -> Result<Vec<ModuleGlobalInitializer>, InvocationShapeError> {
    let stateful = authority.producers.iter().filter_map(|producer| {
        let SemanticBaseProducer::Recorded(recorded) = producer else {
            return None;
        };
        recorded.source.deduce.module_state.map(|_| recorded)
    });
    let stateful = stateful.collect::<Vec<_>>();
    if stateful.is_empty() {
        return Ok(Vec::new());
    }
    if sm_major == 0 || sm_minor >= 10 {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    let target_sm = sm_major
        .checked_mul(10)
        .and_then(|major| major.checked_add(sm_minor))
        .ok_or(InvocationShapeError::SizeOverflow)?;
    let manifest = aot::loaded_manifest_identity();
    if manifest == [0; 32] {
        return Err(InvocationShapeError::MissingLoadedAotAuthority);
    }
    let mut initializers = Vec::new();
    for recorded in stateful {
        let kernel = aot::loaded_kernel_authority(recorded.source.cache_key, sm_major, sm_minor)
            .ok_or(InvocationShapeError::MissingLoadedAotAuthority)?;
        let fields = ResolvedRecordedBuildAuthority::from_embedded(manifest, kernel);
        fields
            .validate(&recorded.source, manifest, target_sm)
            .map_err(|_| InvocationShapeError::LoadedAotAuthorityMismatch)?;
        let (new, effects) = resolve_module_globals(&recorded.source, &fields, &initializers)
            .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
        if effects.len() != 2 {
            return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
        }
        initializers.extend(new);
    }
    Ok(initializers)
}

fn required_base_sources(
    authority: &BaseProducerAuthority,
) -> Result<Vec<RegisteredFixedSourceAuthority>, InvocationShapeError> {
    authority
        .producers
        .iter()
        .filter_map(|producer| match producer {
            SemanticBaseProducer::Recorded(recorded) => recorded.source.deduce.module_state,
            SemanticBaseProducer::NativeBlakeGDirect { .. }
            | SemanticBaseProducer::NativeEcOp { .. } => None,
        })
        .map(|state| {
            registered_pedersen_source(state)
                .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)
        })
        .collect()
}

fn bind_sources(
    required: Vec<RegisteredFixedSourceAuthority>,
    prepared: &[PreparedRegisteredFixedSource],
) -> Result<Vec<LoadedRegisteredFixedSource>, InvocationShapeError> {
    let required = canonical_required_sources(required)?;
    if required.len() != prepared.len() || prepared.len() > 1 {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    required
        .into_iter()
        .zip(prepared)
        .map(|(authority, prepared)| bind_source(authority, prepared))
        .collect()
}

fn canonical_required_sources(
    required: Vec<RegisteredFixedSourceAuthority>,
) -> Result<BTreeSet<RegisteredFixedSourceAuthority>, InvocationShapeError> {
    let canonical = registered_pedersen_source(PedersenTableColumnsAndRowsV1::CANONICAL)
        .map_err(|_| InvocationShapeError::LoadedModuleStateAuthorityMismatch)?;
    let required = required.into_iter().collect::<BTreeSet<_>>();
    if required.iter().any(|source| source != &canonical) || required.len() > 1 {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    Ok(required)
}

fn bind_source(
    authority: RegisteredFixedSourceAuthority,
    prepared: &PreparedRegisteredFixedSource,
) -> Result<LoadedRegisteredFixedSource, InvocationShapeError> {
    if prepared.registered != prepared.canonical {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    let source = &prepared.canonical;
    if source.source_rows != authority.source_rows()
        || source.padded_rows != authority.padded_rows()
        || source.element_bytes != authority.element_bytes()
        || source.generation != PEDERSEN_TABLE_REGISTRATION_GENERATION
        || source.columns.len() != authority.columns().len()
    {
        return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
    }
    for (position, column) in source.columns.iter().enumerate() {
        if column.index != position
            || column.elements != source.padded_rows
            || column.address == 0
            || column.address % source.element_bytes as u64 != 0
        {
            return Err(InvocationShapeError::LoadedModuleStateAuthorityMismatch);
        }
    }
    Ok(LoadedRegisteredFixedSource {
        authority,
        byte_digest: source.byte_digest,
        generation: source.generation,
        column_addresses: source.columns.iter().map(|column| column.address).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled_proof::{ByteRange, ElementRange, ModuleGlobalInitializerAtom};

    fn canonical_authority() -> RegisteredFixedSourceAuthority {
        registered_pedersen_source(PedersenTableColumnsAndRowsV1::CANONICAL).unwrap()
    }

    fn snapshot() -> RegisteredSourceSnapshot {
        let authority = canonical_authority();
        let column_count = authority.columns().len();
        let padded_rows = authority.padded_rows();
        RegisteredSourceSnapshot {
            byte_digest: PedersenTableContentDigest::new([0x51; 32]),
            source_rows: authority.source_rows(),
            padded_rows,
            element_bytes: authority.element_bytes(),
            generation: PEDERSEN_TABLE_REGISTRATION_GENERATION,
            columns: (0..column_count)
                .map(|index| RegisteredColumnSnapshot {
                    index,
                    address: 0x1000 + index as u64 * 0x100,
                    elements: padded_rows,
                })
                .collect(),
        }
    }

    fn prepared() -> PreparedRegisteredFixedSource {
        let snapshot = snapshot();
        PreparedRegisteredFixedSource {
            registered: snapshot.clone(),
            canonical: snapshot,
        }
    }

    fn initializer(
        id: u32,
        module: &[u8],
        source: RegisteredFixedSourceAuthority,
    ) -> ModuleGlobalInitializer {
        let bytes = source.columns().len() * core::mem::size_of::<u64>();
        ModuleGlobalInitializer::new(
            ModuleGlobalInitializerId(id),
            ModuleIdentity::new(module.to_vec()).unwrap(),
            b"g_stwo_wit_pedersen_cols".to_vec(),
            bytes,
            core::mem::align_of::<u64>(),
            true,
            vec![
                ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                    destination: ByteRange::new(0, bytes).unwrap(),
                    source,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn repeated_module_requirements_share_one_table_but_keep_two_relocations() {
        let source = canonical_authority();
        let globals = bind_module_initializers(
            &[
                initializer(0, b"module-a", source.clone()),
                initializer(1, b"module-b", source),
            ],
            &[prepared()],
        )
        .unwrap();
        assert_eq!(globals.sources.len(), 1);
        assert_eq!(globals.relocations.len(), 2);
        assert_eq!(
            globals
                .relocations
                .iter()
                .map(|relocation| relocation.initializer)
                .collect::<Vec<_>>(),
            vec![ModuleGlobalInitializerId(0), ModuleGlobalInitializerId(1)]
        );
        assert_ne!(globals.relocations[0].module, globals.relocations[1].module);
        assert!(globals.relocations.iter().all(|relocation| {
            relocation.symbol.as_ref() == b"g_stwo_wit_pedersen_cols"
                && relocation.destination
                    == ByteRange::new(
                        0,
                        PedersenTableColumnsAndRowsV1::CANONICAL
                            .column_pointers
                            .symbol_bytes as usize,
                    )
                    .unwrap()
                && relocation.source_identity == *canonical_authority().identity()
        }));
        assert_eq!(
            globals.relocations[0].column_addresses,
            globals.relocations[1].column_addresses
        );
        assert_eq!(
            globals.relocations[0].column_addresses.len(),
            PedersenTableColumnsAndRowsV1::CANONICAL.resource.columns as usize
        );
    }

    #[test]
    fn source_inventory_is_closed() {
        let source = canonical_authority();
        let module_initializer = initializer(0, b"module", source);
        assert!(bind_module_initializers(&[module_initializer.clone()], &[]).is_err());
        assert!(
            bind_module_initializers(&[module_initializer.clone()], &[prepared(), prepared()],)
                .is_err()
        );
        assert!(bind_module_initializers(&[], &[prepared()]).is_err());

        let duplicate_id = initializer(0, b"other-module", canonical_authority());
        assert!(
            bind_module_initializers(&[module_initializer, duplicate_id], &[prepared()]).is_err()
        );

        let source = canonical_authority();
        let bytes = source.columns().len() * core::mem::size_of::<u64>();
        let wrong_symbol = ModuleGlobalInitializer::new(
            ModuleGlobalInitializerId(0),
            ModuleIdentity::new(b"module".to_vec()).unwrap(),
            b"wrong_columns".to_vec(),
            bytes,
            core::mem::align_of::<u64>(),
            true,
            vec![
                ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                    destination: ByteRange::new(0, bytes).unwrap(),
                    source,
                },
            ],
        )
        .unwrap();
        assert!(bind_module_initializers(&[wrong_symbol], &[prepared()]).is_err());
    }

    #[test]
    fn canonical_recipe_is_required_not_merely_a_self_consistent_identity() {
        let exact = canonical_authority();
        let columns = exact
            .columns()
            .map(|column| column.to_vec())
            .collect::<Vec<_>>();
        let mut candidates = Vec::new();
        let mut recipe = *exact.recipe_identity();
        recipe[0] ^= 1;
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                recipe,
                exact.source_rows(),
                exact.padded_rows(),
                exact.element_bytes(),
                columns.clone(),
            )
            .unwrap(),
        );
        let mut substituted = columns.clone();
        substituted[0] = b"pedersen_points_wrong".to_vec();
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows(),
                exact.padded_rows(),
                exact.element_bytes(),
                substituted,
            )
            .unwrap(),
        );
        let mut reordered = columns.clone();
        reordered.swap(0, 1);
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows(),
                exact.padded_rows(),
                exact.element_bytes(),
                reordered,
            )
            .unwrap(),
        );
        let mut missing = columns.clone();
        missing.pop();
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows(),
                exact.padded_rows(),
                exact.element_bytes(),
                missing,
            )
            .unwrap(),
        );
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows() - 1,
                exact.padded_rows(),
                exact.element_bytes(),
                columns.clone(),
            )
            .unwrap(),
        );
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows(),
                exact.padded_rows() * 2,
                exact.element_bytes(),
                columns.clone(),
            )
            .unwrap(),
        );
        candidates.push(
            RegisteredFixedSourceAuthority::new(
                *exact.recipe_identity(),
                exact.source_rows(),
                exact.padded_rows(),
                exact.element_bytes() * 2,
                columns,
            )
            .unwrap(),
        );
        for (ordinal, candidate) in candidates.into_iter().enumerate() {
            assert!(
                bind_module_initializers(&[initializer(0, b"module", candidate)], &[prepared()])
                    .is_err(),
                "accepted forged canonical source {ordinal}"
            );
        }
    }

    #[test]
    fn process_registration_pair_and_every_live_field_are_exact() {
        let authority = canonical_authority();
        let exact = prepared();
        bind_sources(vec![authority.clone()], &[exact.clone()]).unwrap();

        let registered_mutations: [fn(&mut RegisteredSourceSnapshot); 9] = [
            |source| source.byte_digest = PedersenTableContentDigest::new([0x99; 32]),
            |source| source.source_rows -= 1,
            |source| source.padded_rows *= 2,
            |source| source.element_bytes *= 2,
            |source| source.generation += 1,
            |source| source.columns[0].index += 1,
            |source| source.columns[0].elements -= 1,
            |source| source.columns[0].address = 0,
            |source| source.columns.swap(0, 1),
        ];
        for (ordinal, mutate) in registered_mutations.into_iter().enumerate() {
            let mut changed = exact.clone();
            mutate(&mut changed.registered);
            assert!(
                bind_sources(vec![authority.clone()], &[changed]).is_err(),
                "accepted registered/canonical mismatch {ordinal}"
            );
        }

        let canonical_mutations: [fn(&mut RegisteredSourceSnapshot); 8] = [
            |source| source.source_rows -= 1,
            |source| source.padded_rows *= 2,
            |source| source.element_bytes *= 2,
            |source| source.generation += 1,
            |source| source.columns.pop().map(drop).unwrap(),
            |source| source.columns[0].elements -= 1,
            |source| source.columns[0].address = 2,
            |source| source.columns[0].index += 1,
        ];
        for (ordinal, mutate) in canonical_mutations.into_iter().enumerate() {
            let mut changed = exact.clone();
            mutate(&mut changed.canonical);
            changed.registered = changed.canonical.clone();
            assert!(
                bind_sources(vec![authority.clone()], &[changed]).is_err(),
                "accepted invalid live source {ordinal}"
            );
        }
    }

    #[test]
    fn direct_read_requires_exact_authority_range_and_live_column() {
        let exact = prepared();
        let authority = canonical_authority();
        let rows = authority.padded_rows();
        let address = exact.canonical.columns[0].address;
        let full = ElementRange::new(0, rows).unwrap();
        let read = RegisteredFixedSourceRead::new(authority.clone(), 0, full).unwrap();
        exact
            .validate_read_column_facts(&read, 0, address, rows)
            .unwrap();

        let mut recipe = *authority.recipe_identity();
        recipe[0] ^= 1;
        let foreign = RegisteredFixedSourceAuthority::new(
            recipe,
            authority.source_rows(),
            rows,
            authority.element_bytes(),
            authority.columns().map(|column| column.to_vec()).collect(),
        )
        .unwrap();
        let foreign_read = RegisteredFixedSourceRead::new(foreign, 0, full).unwrap();
        let partial_read = RegisteredFixedSourceRead::new(
            authority.clone(),
            0,
            ElementRange::new(1, rows).unwrap(),
        )
        .unwrap();
        let cases = [
            (foreign_read, 0, address, rows),
            (partial_read, 0, address, rows - 1),
            (read.clone(), 1, address, rows),
            (
                read.clone(),
                0,
                address + authority.element_bytes() as u64,
                rows,
            ),
            (read.clone(), 0, address, rows - 1),
        ];
        for (ordinal, (read, index, address, elements)) in cases.into_iter().enumerate() {
            assert!(
                exact
                    .validate_read_column_facts(&read, index, address, elements)
                    .is_err(),
                "accepted forged direct read {ordinal}"
            );
        }

        let mut wrong_generation = exact.clone();
        wrong_generation.canonical.generation += 1;
        wrong_generation.registered = wrong_generation.canonical.clone();
        assert!(wrong_generation
            .validate_read_column_facts(&read, 0, address, rows)
            .is_err());
    }
}
