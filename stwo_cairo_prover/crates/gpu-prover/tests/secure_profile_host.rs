use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo_backend_cuda::{
    decommit_workspace_requirements, fri_workspace_requirements, DecommitTreeGeometry,
    DecommitWorkspaceConfig, FriDecommitGeometry, FriWorkspaceConfig,
};

#[test]
fn secure_profile_fri_geometry_and_capacities_are_golden() {
    let pcs = PcsConfig {
        pow_bits: 26,
        fri_config: FriConfig::new(0, 1, 70, 3),
        lifting_log_size: None,
    };
    assert_eq!(pcs.pow_bits, 26);
    assert_eq!(pcs.fri_config.n_queries, 70);
    assert_eq!(pcs.fri_config.n_queries << pcs.fri_config.fold_step, 560);
    let requirements = fri_workspace_requirements(FriWorkspaceConfig {
        fri: pcs.fri_config,
        circle_log_size: 26,
        twiddle_log_size: 25,
    })
    .unwrap();
    let tuples = requirements
        .trees
        .iter()
        .enumerate()
        .map(|(index, tree)| {
            (
                u32::try_from(index).unwrap() * 3,
                tree.evaluation_log_size,
                tree.outgoing_fold_step,
                tree.log_rows_per_leaf,
                tree.layers_bottom_up[0].log_size,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        tuples,
        [
            (0, 26, 3, 2, 24),
            (3, 23, 3, 2, 21),
            (6, 20, 3, 2, 18),
            (9, 17, 3, 2, 15),
            (12, 14, 3, 2, 12),
            (15, 11, 3, 2, 9),
            (18, 8, 3, 2, 6),
            (21, 5, 3, 2, 3),
            (24, 2, 1, 0, 2),
        ]
    );
    assert_eq!(requirements.last_layer_log_size, 1);
    assert_eq!(requirements.twiddle_words, 1 << 25);
    assert_eq!(requirements.evaluation_ping_words, 4 << 25);
    assert_eq!(requirements.evaluation_pong_words, 4 << 25);
    assert_eq!(requirements.coordinate_pointer_words, 8);

    let decommit = decommit_workspace_requirements(DecommitWorkspaceConfig {
        query_log_size: 26,
        n_queries: u32::try_from(pcs.fri_config.n_queries).unwrap(),
        trees: requirements
            .trees
            .iter()
            .enumerate()
            .map(|(index, tree)| {
                DecommitTreeGeometry::Fri(FriDecommitGeometry {
                    fri_tree_index: u32::try_from(index).unwrap(),
                    evaluation_log_size: tree.evaluation_log_size,
                    cumulative_fold: u32::try_from(index).unwrap() * 3,
                    outgoing_fold_step: tree.outgoing_fold_step,
                    log_rows_per_leaf: tree.log_rows_per_leaf,
                })
            })
            .collect(),
    })
    .unwrap();
    assert_eq!(decommit.config.n_queries, 70);
    assert_eq!(decommit.expanded_position_words, 560);
    assert_eq!(decommit.walk_query_words, 560);
    assert_eq!(
        decommit
            .trees
            .iter()
            .filter_map(|tree| match tree {
                stwo_backend_cuda::DecommitTreeRequirements::Fri(fri) => {
                    Some(fri.max_expanded_positions)
                }
                stwo_backend_cuda::DecommitTreeRequirements::Trace(_) => None,
            })
            .max(),
        Some(560)
    );
}
