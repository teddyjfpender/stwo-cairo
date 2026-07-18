use super::*;

pub(super) fn validate_receipt(
    arena: &ProofArenaPlan,
    prelude: &LoweredCompositionPrelude,
    lowered: &LoweredCompositionWaves,
) -> Result<(), InvocationShapeError> {
    lowered
        .authority
        .validate_against(arena)
        .map_err(|_| InvocationShapeError::InvalidCompositionAuthority)?;
    if lowered.waves.len() != lowered.authority.waves().len()
        || lowered.waves.iter().enumerate().any(|(index, wave)| {
            wave.shard.wave_index() != index
                || wave.shard.effect() != wave.effect.id()
                || wave.shard.partition().kind()
                    == &crate::compiled_proof::PartitionAuthorityKind::Monolithic
        })
        || receipt_digest(&lowered.authority, prelude, &lowered.waves)? != lowered.digest
    {
        return Err(InvocationShapeError::InvalidCompositionBinding);
    }
    Ok(())
}

pub(super) fn receipt_digest(
    authority: &CompositionExecutionAuthority,
    prelude: &LoweredCompositionPrelude,
    waves: &[LoweredCompositionWave],
) -> Result<[u8; 32], InvocationShapeError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    hasher.update(&authority.identity());
    hasher.update(&prelude.digest());
    hasher.update(
        &u64::try_from(waves.len())
            .map_err(|_| InvocationShapeError::SizeOverflow)?
            .to_le_bytes(),
    );
    for wave in waves {
        hasher.update(&wave.operation_ordinal.to_le_bytes());
        hasher.update(&wave.operation.identity);
        hasher.update(wave.effect.id().as_bytes());
        hasher.update(
            wave.invocation
                .contract_id()
                .map_err(|_| InvocationShapeError::InvalidCompositionBinding)?
                .as_bytes(),
        );
        hasher.update(wave.shard.digest());
    }
    Ok(*hasher.finalize().as_bytes())
}
