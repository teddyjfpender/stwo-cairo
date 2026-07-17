use super::*;
use crate::compiled_proof::{
    ByteRange, ModuleGlobalEffect, ModuleGlobalInitializer, ModuleGlobalInitializerAtom,
    ModuleGlobalInitializerId, ModuleIdentity, RegisteredFixedSourceAuthority,
};

const PEDERSEN_CONTENT_DOMAIN: &[u8] = b"stwo-cairo.recorded-base.pedersen-registered-content.v1\0";

pub(in super::super) fn resolve(
    source: &RecordedWitnessInvocationShape,
    fields: &ResolvedRecordedBuildAuthority,
    existing: &[ModuleGlobalInitializer],
) -> Result<(Vec<ModuleGlobalInitializer>, Vec<ModuleGlobalEffect>), ()> {
    let Some(state) = source.deduce.module_state else {
        return Ok((Vec::new(), Vec::new()));
    };
    let module = compiled_module(fields)?;
    let columns = existing.iter().find(|initializer| {
        initializer.module() == &module
            && initializer.symbol() == state.column_pointers.symbol.as_bytes()
    });
    let rows = existing.iter().find(|initializer| {
        initializer.module() == &module && initializer.symbol() == state.row_count.symbol.as_bytes()
    });
    match (columns, rows) {
        (None, None) => for_source(
            source,
            fields,
            ModuleGlobalInitializerId(u32::try_from(existing.len()).map_err(|_| ())?),
        ),
        (Some(columns), Some(rows)) => {
            let (expected, effects) = for_source(source, fields, columns.id())?;
            if expected != vec![columns.clone(), rows.clone()] {
                return Err(());
            }
            Ok((Vec::new(), effects))
        }
        _ => Err(()),
    }
}

pub(super) fn validate_effect(
    source: &RecordedWitnessInvocationShape,
    module: &ModuleIdentity,
    effect: &EffectContract,
    initializers: &[ModuleGlobalInitializer],
) -> Result<(), ()> {
    let Some(state) = source.deduce.module_state else {
        return effect.module_globals().is_empty().then_some(()).ok_or(());
    };
    let globals = effect.module_globals();
    if globals.len() != 2 {
        return Err(());
    }
    let (expected_initializers, expected_globals) =
        for_state(module.clone(), state, globals[0].initializer)?;
    if globals != expected_globals {
        return Err(());
    }
    for expected in expected_initializers {
        let actual = initializers
            .get(expected.id().0 as usize)
            .filter(|actual| actual.id() == expected.id())
            .ok_or(())?;
        if actual != &expected {
            return Err(());
        }
    }
    Ok(())
}

fn for_source(
    source: &RecordedWitnessInvocationShape,
    fields: &ResolvedRecordedBuildAuthority,
    first_id: ModuleGlobalInitializerId,
) -> Result<(Vec<ModuleGlobalInitializer>, Vec<ModuleGlobalEffect>), ()> {
    let Some(state) = source.deduce.module_state else {
        return Ok((Vec::new(), Vec::new()));
    };
    for_state(compiled_module(fields)?, state, first_id)
}

fn for_state(
    module: ModuleIdentity,
    state: recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
    first_id: ModuleGlobalInitializerId,
) -> Result<(Vec<ModuleGlobalInitializer>, Vec<ModuleGlobalEffect>), ()> {
    state.validate_exact().map_err(|_| ())?;
    let columns_id = first_id;
    let rows_id = ModuleGlobalInitializerId(first_id.0.checked_add(1).ok_or(())?);
    let columns_bytes = usize::try_from(state.column_pointers.symbol_bytes).map_err(|_| ())?;
    let rows_bytes = usize::try_from(state.row_count.symbol_bytes).map_err(|_| ())?;
    let columns_range = ByteRange::new(0, columns_bytes).ok_or(())?;
    let rows_range = ByteRange::new(0, rows_bytes).ok_or(())?;
    let columns = ModuleGlobalInitializer::new(
        columns_id,
        module.clone(),
        state.column_pointers.symbol.as_bytes().to_vec(),
        columns_bytes,
        usize::try_from(state.column_pointers.alignment_bytes).map_err(|_| ())?,
        true,
        vec![
            ModuleGlobalInitializerAtom::RegisteredFixedSourceColumnAddresses {
                destination: columns_range,
                source: registered_pedersen_source(state)?,
            },
        ],
    )
    .map_err(|_| ())?;
    let rows = ModuleGlobalInitializer::new(
        rows_id,
        module,
        state.row_count.symbol.as_bytes().to_vec(),
        rows_bytes,
        usize::try_from(state.row_count.alignment_bytes).map_err(|_| ())?,
        true,
        vec![ModuleGlobalInitializerAtom::Literal {
            destination: rows_range,
            bytes: state
                .row_count
                .value
                .to_le_bytes()
                .to_vec()
                .into_boxed_slice(),
        }],
    )
    .map_err(|_| ())?;
    Ok((
        vec![columns, rows],
        vec![
            ModuleGlobalEffect {
                initializer: columns_id,
                bytes: columns_range,
            },
            ModuleGlobalEffect {
                initializer: rows_id,
                bytes: rows_range,
            },
        ],
    ))
}

pub(in super::super) fn registered_pedersen_source(
    state: recorded_deduce_authority::PedersenTableColumnsAndRowsV1,
) -> Result<RegisteredFixedSourceAuthority, ()> {
    state.validate_exact().map_err(|_| ())?;
    let resource = state.resource;
    let mut hasher = blake3::Hasher::new();
    hasher.update(PEDERSEN_CONTENT_DOMAIN);
    hash_bytes(&mut hasher, resource.source_recipe.as_bytes())?;
    hash_bytes(&mut hasher, resource.column_identity_prefix.as_bytes())?;
    for value in [
        resource.first_column,
        resource.columns,
        resource.coordinate_limbs,
        resource.semantic_real_rows,
        resource.registered_source_rows,
        resource.registered_padded_rows,
        resource.uploader_extra_rows,
        resource.element_bytes,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&[match resource.semantic_padding {
        recorded_deduce_authority::PedersenSemanticPaddingV1::RepeatRowZero => 1,
    }]);
    hasher.update(&[match resource.uploader_extra_padding {
        recorded_deduce_authority::PedersenUploaderExtraPaddingV1::ZeroFill => 1,
    }]);
    hasher.update(&[match resource.content_authority {
        recorded_deduce_authority::PedersenContentAuthorityV1::RequiredFromCanonicalRegistration => {
            1
        }
    }]);
    let columns = (0..resource.columns)
        .map(|offset| {
            resource
                .first_column
                .checked_add(offset)
                .map(|column| format!("{}{}", resource.column_identity_prefix, column).into_bytes())
                .ok_or(())
        })
        .collect::<Result<Vec<_>, _>>()?;
    RegisteredFixedSourceAuthority::new(
        *hasher.finalize().as_bytes(),
        usize::try_from(resource.registered_source_rows).map_err(|_| ())?,
        usize::try_from(resource.registered_padded_rows).map_err(|_| ())?,
        usize::try_from(resource.element_bytes).map_err(|_| ())?,
        columns,
    )
    .map_err(|_| ())
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) -> Result<(), ()> {
    hasher.update(&u64::try_from(bytes.len()).map_err(|_| ())?.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}
