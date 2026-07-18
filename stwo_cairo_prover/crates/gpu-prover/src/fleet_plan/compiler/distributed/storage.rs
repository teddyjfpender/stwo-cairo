//! Conservative storage materialization for the partitioned compiler.

use std::collections::BTreeMap;

use super::*;

pub(super) fn compile_partitioned_storage(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    owners: &[FleetOwnerPlacement],
    replicas: &[FleetReplicaPlacement],
    output: &[OutputBinding],
) -> Result<(Vec<StorageDesc>, Vec<FleetStoragePlacement>, StorageId), FleetCompileError> {
    let mut storages = Vec::new();
    let mut bindings = Vec::new();
    let mut locations = BTreeMap::<(WorkerId, ValueVersion), Vec<ElementRange>>::new();
    for (worker, value) in owners
        .iter()
        .map(|owner| (owner.worker, owner.value))
        .chain(
            replicas
                .iter()
                .map(|replica| (replica.worker, replica.value)),
        )
    {
        locations
            .entry((worker, value.version))
            .or_default()
            .push(value.elements);
    }
    allocate_statement_lineages(
        compiled,
        coordinator,
        &mut locations,
        &mut storages,
        &mut bindings,
    )?;
    for ((worker, version), mut remainder) in locations {
        if worker == coordinator {
            for reserved in output
                .iter()
                .filter(|binding| binding.value.version == version)
                .map(|binding| binding.value.elements)
            {
                remainder = subtract_all(remainder, reserved);
            }
        }
        if remainder.is_empty() {
            continue;
        }
        remainder.sort_unstable_by_key(|range| (range.start, range.end));
        if remainder.windows(2).any(|pair| pair[0].end > pair[1].start) {
            return Err(FleetCompileError::InvalidOwnership(version));
        }
        let desc = compiled
            .value(version)
            .ok_or(FleetCompileError::InvalidOwnership(version))?;
        allocate_affine_locations(&mut storages, &mut bindings, worker, desc, &remainder)?;
    }

    let output_storage =
        StorageId(u32::try_from(storages.len()).map_err(|_| FleetCompileError::SizeOverflow)?);
    let alignment = compiled
        .output()
        .fragments
        .iter()
        .filter_map(|fragment| compiled.value(fragment.source.version))
        .map(|value| value.alignment)
        .max()
        .unwrap_or(core::mem::align_of::<u32>())
        .max(core::mem::align_of::<u32>());
    storages.push(StorageDesc {
        id: output_storage,
        worker: coordinator,
        bytes: compiled
            .output()
            .layout
            .total_words
            .checked_mul(core::mem::size_of::<u32>())
            .ok_or(FleetCompileError::SizeOverflow)?,
        alignment_bytes: alignment,
    });
    for binding in output {
        bindings.push(FleetStoragePlacement {
            storage: output_storage,
            value: binding.value,
            offset_bytes: binding.offset_bytes,
        });
    }
    Ok((storages, bindings, output_storage))
}

fn allocate_statement_lineages(
    compiled: &CompiledProof,
    coordinator: WorkerId,
    locations: &mut BTreeMap<(WorkerId, ValueVersion), Vec<ElementRange>>,
    storages: &mut Vec<StorageDesc>,
    bindings: &mut Vec<FleetStoragePlacement>,
) -> Result<(), FleetCompileError> {
    let reuses = statement_host_reuses(compiled)?;
    let components = compile_ingress_components(compiled.values().len(), &reuses)?;
    let component_count = components
        .iter()
        .flatten()
        .max()
        .map_or(0, |value| value + 1);
    for component in 0..component_count {
        let versions = components
            .iter()
            .enumerate()
            .filter_map(|(version, candidate)| (*candidate == Some(component)).then_some(version))
            .map(|version| {
                u32::try_from(version)
                    .map(ValueVersion)
                    .map_err(|_| FleetCompileError::SizeOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first = versions
            .first()
            .and_then(|version| compiled.value(*version))
            .ok_or(FleetCompileError::InvalidSemanticSchedule)?;
        let storage =
            StorageId(u32::try_from(storages.len()).map_err(|_| FleetCompileError::SizeOverflow)?);
        storages.push(StorageDesc {
            id: storage,
            worker: coordinator,
            bytes: first
                .layout
                .logical_bytes()
                .map_err(|_| FleetCompileError::SizeOverflow)?,
            alignment_bytes: first.alignment,
        });
        for version in versions {
            let value = compiled
                .value(version)
                .ok_or(FleetCompileError::InvalidOwnership(version))?;
            let full = full_range(value)?;
            if locations.remove(&(coordinator, version)) != Some(vec![full]) {
                return Err(FleetCompileError::InvalidOwnership(version));
            }
            bindings.push(FleetStoragePlacement {
                storage,
                value: ValueRange {
                    version,
                    elements: full,
                },
                offset_bytes: 0,
            });
        }
    }
    Ok(())
}

fn allocate_affine_locations(
    storages: &mut Vec<StorageDesc>,
    bindings: &mut Vec<FleetStoragePlacement>,
    worker: WorkerId,
    value: &ValueDesc,
    ranges: &[ElementRange],
) -> Result<(), FleetCompileError> {
    let storage =
        StorageId(u32::try_from(storages.len()).map_err(|_| FleetCompileError::SizeOverflow)?);
    let base = ranges
        .first()
        .ok_or(FleetCompileError::InvalidOwnership(value.version))?
        .start;
    let end = ranges
        .last()
        .ok_or(FleetCompileError::InvalidOwnership(value.version))?
        .end;
    storages.push(StorageDesc {
        id: storage,
        worker,
        bytes: end
            .checked_sub(base)
            .and_then(|elements| elements.checked_mul(value.layout.element.bytes))
            .ok_or(FleetCompileError::SizeOverflow)?,
        alignment_bytes: value.alignment,
    });
    for &elements in ranges {
        let offset_bytes = elements
            .start
            .checked_sub(base)
            .and_then(|elements| elements.checked_mul(value.layout.element.bytes))
            .ok_or(FleetCompileError::SizeOverflow)?;
        bindings.push(FleetStoragePlacement {
            storage,
            value: ValueRange {
                version: value.version,
                elements,
            },
            offset_bytes,
        });
    }
    Ok(())
}

fn subtract_all(ranges: Vec<ElementRange>, reserved: ElementRange) -> Vec<ElementRange> {
    let mut output = Vec::new();
    for range in ranges {
        if !range.overlaps(reserved) {
            output.push(range);
            continue;
        }
        if range.start < reserved.start {
            output.push(ElementRange {
                start: range.start,
                end: reserved.start.min(range.end),
            });
        }
        if reserved.end < range.end {
            output.push(ElementRange {
                start: reserved.end.max(range.start),
                end: range.end,
            });
        }
    }
    output
}
