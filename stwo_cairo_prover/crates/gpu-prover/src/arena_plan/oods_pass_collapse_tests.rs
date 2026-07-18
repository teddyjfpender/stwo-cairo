use super::*;

fn config() -> OodsWorkspaceConfig {
    OodsWorkspaceConfig {
        lifting_log_size: 20,
        mask_log_size: 18,
    }
}

#[test]
fn replacement_selects_exact_program_and_legacy_never_does() {
    let masks = [0, 1, -1];
    let columns = [OodsColumnTopology::evaluation_signed_offsets(18, &masks)];
    let selected = select_oods_pass_collapse(ResidentBackend::ReplacementV1, config(), &columns)
        .unwrap()
        .expect("evaluation-backed ReplacementV1 OODS must select collapse");
    assert_eq!(
        selected.ordinary_requirements(),
        &oods_workspace_requirements(config(), &columns).unwrap()
    );
    validate_oods_pass_collapse_selection(
        ResidentBackend::ReplacementV1,
        config(),
        &columns,
        Some(&selected),
    )
    .unwrap();
    let cached = BufferLifetime::new(ProofEpoch::Ingest, ProofEpoch::Assemble).unwrap();
    let scratch = BufferLifetime::at(ProofEpoch::Oods);
    assert_eq!(
        oods_barycentric_scales_lifetime(Some(&selected), cached, scratch),
        cached,
        "collapsed descriptor offsets must survive every captured replay"
    );

    assert!(
        select_oods_pass_collapse(ResidentBackend::LegacyResident, config(), &columns)
            .unwrap()
            .is_none()
    );
    validate_oods_pass_collapse_selection(
        ResidentBackend::LegacyResident,
        config(),
        &columns,
        None,
    )
    .unwrap();
    assert_eq!(
        oods_barycentric_scales_lifetime(None, cached, scratch),
        scratch,
        "ordinary OODS keeps barycentric scales as epoch-local scratch"
    );
}

#[test]
fn selection_rejects_backend_topology_and_presence_drift() {
    let masks = [0, 1];
    let columns = [OodsColumnTopology::evaluation_signed_offsets(18, &masks)];
    let selected = select_oods_pass_collapse(ResidentBackend::ReplacementV1, config(), &columns)
        .unwrap()
        .unwrap();

    assert!(matches!(
        validate_oods_pass_collapse_selection(
            ResidentBackend::LegacyResident,
            config(),
            &columns,
            Some(&selected),
        ),
        Err(OodsPassCollapseSelectionError::SelectionDrift { .. })
    ));
    assert!(matches!(
        validate_oods_pass_collapse_selection(
            ResidentBackend::ReplacementV1,
            config(),
            &columns,
            None,
        ),
        Err(OodsPassCollapseSelectionError::SelectionDrift { .. })
    ));

    let reordered_masks = [1, 0];
    let reordered = [OodsColumnTopology::evaluation_signed_offsets(
        18,
        &reordered_masks,
    )];
    assert!(matches!(
        validate_oods_pass_collapse_selection(
            ResidentBackend::ReplacementV1,
            config(),
            &reordered,
            Some(&selected),
        ),
        Err(OodsPassCollapseSelectionError::Identity(
            OodsPassCollapseError::ProgramIdentity
        ))
    ));
}

#[test]
fn replacement_falls_back_only_when_there_is_no_evaluation_pass() {
    let masks = [0, 1];
    let columns = [OodsColumnTopology::coefficient_signed_offsets(
        18, 20, &masks,
    )];
    assert!(
        select_oods_pass_collapse(ResidentBackend::ReplacementV1, config(), &columns)
            .unwrap()
            .is_none()
    );
    validate_oods_pass_collapse_selection(ResidentBackend::ReplacementV1, config(), &columns, None)
        .unwrap();

    assert!(matches!(
        select_oods_pass_collapse(
            ResidentBackend::ReplacementV1,
            OodsWorkspaceConfig {
                lifting_log_size: 0,
                mask_log_size: 0,
            },
            &columns,
        ),
        Err(OodsPassCollapseSelectionError::Compile(
            OodsPassCollapseError::Oods(PreparedOodsError::InvalidLiftingLogSize(0))
        ))
    ));
}
