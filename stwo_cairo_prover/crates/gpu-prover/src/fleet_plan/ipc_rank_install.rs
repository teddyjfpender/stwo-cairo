//! Rank-local live CUDA-IPC installation for one exact two-worker plan.
//!
//! CUDA context authority never leaves this process. Copyable descriptor and
//! acknowledgement statements are transport payloads only; the controller must
//! authenticate its rank channel before using them.

use stwo_backend_cuda::{
    cuda_context_device_identity, CudaDeviceUuid, CudaExecContext, CudaIpcExchangeImport,
    CudaIpcExchangeOwner, IpcExchangeDescriptor, IpcExchangeError, IpcExchangeInstallDomain,
    IpcExchangeKey, IpcExchangePhase, IpcExchangePhaseReceipt,
};

use super::ipc_cursor::{FleetIpcRankPhaseStatement, FleetIpcRuntimeRosterBinding};
use super::*;

const DESCRIPTOR_DIGEST_TAG: &[u8] = b"stwo-cairo.fleet-ipc-descriptors.v1\0";

#[derive(Debug)]
pub(crate) enum FleetIpcRankInstallError {
    Cursor(FleetIpcCursorError),
    Cuda(IpcExchangeError),
    ExpectedTwoWorkers(usize),
    MissingRuntimeDevice(WorkerId),
    DescriptorStatementOrder {
        expected: WorkerId,
        actual: WorkerId,
    },
    DescriptorStatementMismatch(WorkerId),
    DescriptorCount {
        expected: usize,
        actual: usize,
    },
    DescriptorEdgeOrder {
        expected: u64,
        actual: u64,
    },
    DescriptorKeyMismatch(u64),
    LocalDescriptorMismatch(u64),
    InvalidLocalPhase(u64),
}

impl core::fmt::Display for FleetIpcRankInstallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid rank-local fleet IPC install: {self:?}")
    }
}

impl std::error::Error for FleetIpcRankInstallError {}

impl From<FleetIpcCursorError> for FleetIpcRankInstallError {
    fn from(value: FleetIpcCursorError) -> Self {
        Self::Cursor(value)
    }
}

impl From<IpcExchangeError> for FleetIpcRankInstallError {
    fn from(value: IpcExchangeError) -> Self {
        Self::Cuda(value)
    }
}

/// Owner descriptors emitted by one live rank process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FleetIpcRankDescriptorStatement {
    plan_identity: [u8; 32],
    worker: WorkerId,
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    descriptors: Box<[IpcExchangeDescriptor]>,
}

impl FleetIpcRankDescriptorStatement {
    pub(crate) const fn worker(&self) -> WorkerId {
        self.worker
    }

    pub(crate) fn descriptors(&self) -> &[IpcExchangeDescriptor] {
        &self.descriptors
    }

    #[cfg(test)]
    pub(super) fn test_only(
        plan_identity: [u8; 32],
        worker: WorkerId,
        proof_generation: u64,
        install_domain: IpcExchangeInstallDomain,
        descriptors: Vec<IpcExchangeDescriptor>,
    ) -> Self {
        Self {
            plan_identity,
            worker,
            proof_generation,
            install_domain,
            descriptors: descriptors.into_boxed_slice(),
        }
    }
}

/// Exact ordered owner-descriptor bundle assembled by the controller.
///
/// This is transport structure, not proof that either peer opened a handle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FleetIpcDescriptorBundle {
    plan_identity: [u8; 32],
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    descriptor_digest: [u8; 32],
    descriptors: Box<[IpcExchangeDescriptor]>,
}

impl FleetIpcDescriptorBundle {
    pub(crate) fn join_two_rank(
        view: &FleetRuntimeView,
        statements: [&FleetIpcRankDescriptorStatement; 2],
    ) -> Result<Self, FleetIpcRankInstallError> {
        require_two_workers(view)?;
        for (ordinal, statement) in statements.iter().enumerate() {
            let expected = WorkerId(ordinal as u16);
            if statement.worker != expected {
                return Err(FleetIpcRankInstallError::DescriptorStatementOrder {
                    expected,
                    actual: statement.worker,
                });
            }
            if statement.plan_identity != view.plan_identity()
                || statement.proof_generation != statements[0].proof_generation
                || statement.install_domain != statements[0].install_domain
            {
                return Err(FleetIpcRankInstallError::DescriptorStatementMismatch(
                    statement.worker,
                ));
            }
        }

        let mut descriptors = statements
            .iter()
            .flat_map(|statement| statement.descriptors.iter().cloned())
            .collect::<Vec<_>>();
        descriptors.sort_unstable_by_key(|descriptor| descriptor.key().edge_id());
        validate_descriptor_geometry(
            view,
            statements[0].proof_generation,
            statements[0].install_domain,
            &descriptors,
        )?;
        for descriptor in &descriptors {
            let owner = WorkerId(descriptor.key().owner_rank() as u16);
            let statement = &statements[usize::from(owner.0)];
            if statement.worker != owner
                || !statement
                    .descriptors
                    .iter()
                    .any(|candidate| candidate.encode() == descriptor.encode())
            {
                return Err(FleetIpcRankInstallError::DescriptorKeyMismatch(
                    descriptor.key().edge_id(),
                ));
            }
        }

        let descriptor_digest = descriptor_digest(
            view.plan_identity(),
            statements[0].proof_generation,
            statements[0].install_domain,
            &descriptors,
        );
        Ok(Self {
            plan_identity: view.plan_identity(),
            proof_generation: statements[0].proof_generation,
            install_domain: statements[0].install_domain,
            descriptor_digest,
            descriptors: descriptors.into_boxed_slice(),
        })
    }

    pub(crate) fn descriptors(&self) -> &[IpcExchangeDescriptor] {
        &self.descriptors
    }

    pub(crate) const fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor_digest
    }
}

struct LocalOwner<'context> {
    edge: u64,
    exchange: CudaIpcExchangeOwner<'context>,
    descriptor: Option<IpcExchangeDescriptor>,
}

struct LocalImport<'context> {
    edge: u64,
    exchange: CudaIpcExchangeImport<'context>,
}

/// First rank-local install phase: live owner allocations and exported handles.
#[must_use = "install the incoming peer descriptors before running the proof"]
pub(crate) struct FleetIpcRankOwnerExports<'context> {
    view_identity: [u8; 32],
    proof_generation: u64,
    binding: FleetIpcRuntimeRosterBinding,
    keys: Box<[IpcExchangeKey]>,
    owners: Vec<LocalOwner<'context>>,
    statement: FleetIpcRankDescriptorStatement,
}

impl<'context> FleetIpcRankOwnerExports<'context> {
    pub(crate) fn create(
        view: &FleetRuntimeView,
        proof_generation: u64,
        controller_install_nonce: [u8; 32],
        local_worker: WorkerId,
        context: &'context CudaExecContext,
        roster: &[(WorkerId, CudaDeviceUuid)],
    ) -> Result<Self, FleetIpcRankInstallError> {
        require_two_workers(view)?;
        let identity = cuda_context_device_identity(context)?;
        let binding = FleetIpcRuntimeRosterBinding::bind_local(
            view,
            controller_install_nonce,
            local_worker,
            identity,
            roster,
        )?;
        let keys = derive_keys(view, proof_generation, &binding)?;

        let mut owners = Vec::new();
        for (&span, &key) in view.spans().iter().zip(&keys) {
            if span.owner == local_worker {
                owners.push(LocalOwner {
                    edge: span.edge_ordinal,
                    exchange: CudaIpcExchangeOwner::new(context, key)?,
                    descriptor: None,
                });
            }
        }
        let mut descriptors = Vec::with_capacity(owners.len());
        for owner in &mut owners {
            let descriptor = owner.exchange.export()?;
            descriptors.push(descriptor.clone());
            owner.descriptor = Some(descriptor);
        }
        let statement = FleetIpcRankDescriptorStatement {
            plan_identity: view.plan_identity(),
            worker: local_worker,
            proof_generation,
            install_domain: binding.install_domain()?,
            descriptors: descriptors.into_boxed_slice(),
        };
        Ok(Self {
            view_identity: view.plan_identity(),
            proof_generation,
            binding,
            keys: keys.into_boxed_slice(),
            owners,
            statement,
        })
    }

    pub(crate) const fn statement(&self) -> &FleetIpcRankDescriptorStatement {
        &self.statement
    }

    pub(crate) fn install_imports(
        self,
        view: &FleetRuntimeView,
        context: &'context CudaExecContext,
        bundle: &FleetIpcDescriptorBundle,
    ) -> Result<FleetIpcRankInstalled<'context>, FleetIpcRankInstallError> {
        if self.view_identity != view.plan_identity()
            || bundle.plan_identity != self.view_identity
            || bundle.proof_generation != self.proof_generation
            || bundle.install_domain != self.binding.install_domain()?
        {
            return Err(FleetIpcRankInstallError::DescriptorStatementMismatch(
                self.binding.local_worker(),
            ));
        }
        validate_descriptor_keys(&self.keys, bundle.descriptors())?;
        for owner in &self.owners {
            let descriptor = &bundle.descriptors[owner.edge as usize];
            if owner
                .descriptor
                .as_ref()
                .map_or(true, |local| descriptor.encode() != local.encode())
            {
                return Err(FleetIpcRankInstallError::LocalDescriptorMismatch(
                    owner.edge,
                ));
            }
        }

        let local_worker = self.binding.local_worker();
        let mut imports = Vec::new();
        for (&span, descriptor) in view.spans().iter().zip(bundle.descriptors()) {
            if span.peer == local_worker {
                imports.push(LocalImport {
                    edge: span.edge_ordinal,
                    exchange: CudaIpcExchangeImport::open(
                        context,
                        self.keys[span.edge_ordinal as usize],
                        descriptor.clone(),
                    )?,
                });
            }
        }
        let acknowledgement = FleetIpcRankInstallAcknowledgement {
            plan_identity: self.view_identity,
            worker: local_worker,
            proof_generation: self.proof_generation,
            install_domain: bundle.install_domain,
            descriptor_digest: bundle.descriptor_digest,
            owner_count: self.owners.len() as u32,
            import_count: imports.len() as u32,
        };
        Ok(FleetIpcRankInstalled {
            plan_identity: self.view_identity,
            proof_generation: self.proof_generation,
            binding: self.binding,
            keys: self.keys,
            owners: self.owners,
            imports,
            acknowledgement,
        })
    }
}

/// Rank-local live CUDA objects retained for the proof attempt.
#[must_use = "retain the live exchanges until the proof and close handshake finish"]
pub(crate) struct FleetIpcRankInstalled<'context> {
    plan_identity: [u8; 32],
    proof_generation: u64,
    binding: FleetIpcRuntimeRosterBinding,
    keys: Box<[IpcExchangeKey]>,
    owners: Vec<LocalOwner<'context>>,
    imports: Vec<LocalImport<'context>>,
    acknowledgement: FleetIpcRankInstallAcknowledgement,
}

impl<'context> FleetIpcRankInstalled<'context> {
    pub(crate) const fn acknowledgement(&self) -> FleetIpcRankInstallAcknowledgement {
        self.acknowledgement
    }

    pub(crate) fn owner_mut<'borrow>(
        &'borrow mut self,
        edge: u64,
    ) -> Option<&'borrow mut CudaIpcExchangeOwner<'context>> {
        self.owners
            .iter_mut()
            .find(|owner| owner.edge == edge)
            .map(|owner| &mut owner.exchange)
    }

    pub(crate) fn import_mut<'borrow>(
        &'borrow mut self,
        edge: u64,
    ) -> Option<&'borrow mut CudaIpcExchangeImport<'context>> {
        self.imports
            .iter_mut()
            .find(|import| import.edge == edge)
            .map(|import| &mut import.exchange)
    }

    pub(crate) fn phase_statement(
        &self,
        receipt: IpcExchangePhaseReceipt,
    ) -> Result<FleetIpcRankPhaseStatement, FleetIpcRankInstallError> {
        let key = receipt.key();
        let edge = key.edge_id();
        if self.keys.get(edge as usize) != Some(&key) {
            return Err(FleetIpcRankInstallError::DescriptorKeyMismatch(edge));
        }
        let local_worker = self.binding.local_worker();
        let expected_rank = match receipt.phase() {
            IpcExchangePhase::Published | IpcExchangePhase::Reclaimed => key.owner_rank(),
            IpcExchangePhase::Consumed | IpcExchangePhase::Armed => key.peer_rank(),
        };
        if expected_rank != u32::from(local_worker.0) {
            return Err(FleetIpcRankInstallError::InvalidLocalPhase(edge));
        }
        Ok(FleetIpcRankPhaseStatement::from_backend(
            self.plan_identity,
            local_worker,
            receipt,
        ))
    }
}

/// Transportable acknowledgement that this rank retained all of its live
/// owner/import objects. It is not controller-wide CUDA authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FleetIpcRankInstallAcknowledgement {
    plan_identity: [u8; 32],
    worker: WorkerId,
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    descriptor_digest: [u8; 32],
    owner_count: u32,
    import_count: u32,
}

impl FleetIpcRankInstallAcknowledgement {
    pub(crate) const fn worker(&self) -> WorkerId {
        self.worker
    }

    pub(crate) const fn descriptor_digest(&self) -> [u8; 32] {
        self.descriptor_digest
    }
}

fn require_two_workers(view: &FleetRuntimeView) -> Result<(), FleetIpcRankInstallError> {
    let actual = view.exchange_reserves().len();
    if actual == 2 {
        Ok(())
    } else {
        Err(FleetIpcRankInstallError::ExpectedTwoWorkers(actual))
    }
}

fn derive_keys(
    view: &FleetRuntimeView,
    proof_generation: u64,
    binding: &FleetIpcRuntimeRosterBinding,
) -> Result<Vec<IpcExchangeKey>, FleetIpcRankInstallError> {
    let domain = binding.install_domain()?;
    view.spans()
        .iter()
        .map(|span| {
            let owner = binding
                .device(span.owner)
                .ok_or(FleetIpcRankInstallError::MissingRuntimeDevice(span.owner))?;
            let peer = binding
                .device(span.peer)
                .ok_or(FleetIpcRankInstallError::MissingRuntimeDevice(span.peer))?;
            IpcExchangeKey::new(
                span.edge_ordinal,
                u32::from(span.owner.0),
                u32::from(span.peer.0),
                owner,
                peer,
                domain,
                span.logical_bytes(),
                proof_generation,
            )
            .map_err(FleetIpcRankInstallError::Cuda)
        })
        .collect()
}

fn validate_descriptor_geometry(
    view: &FleetRuntimeView,
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    descriptors: &[IpcExchangeDescriptor],
) -> Result<(), FleetIpcRankInstallError> {
    if descriptors.len() != view.spans().len() {
        return Err(FleetIpcRankInstallError::DescriptorCount {
            expected: view.spans().len(),
            actual: descriptors.len(),
        });
    }
    for (expected, (&span, descriptor)) in view.spans().iter().zip(descriptors).enumerate() {
        let expected = expected as u64;
        let key = descriptor.key();
        if key.edge_id() != expected {
            return Err(FleetIpcRankInstallError::DescriptorEdgeOrder {
                expected,
                actual: key.edge_id(),
            });
        }
        if key.edge_id() != span.edge_ordinal
            || key.owner_rank() != u32::from(span.owner.0)
            || key.peer_rank() != u32::from(span.peer.0)
            || key.logical_bytes() != span.logical_bytes()
            || key.initial_generation() != proof_generation
            || key.install_domain() != install_domain
        {
            return Err(FleetIpcRankInstallError::DescriptorKeyMismatch(expected));
        }
    }
    Ok(())
}

fn validate_descriptor_keys(
    keys: &[IpcExchangeKey],
    descriptors: &[IpcExchangeDescriptor],
) -> Result<(), FleetIpcRankInstallError> {
    if keys.len() != descriptors.len() {
        return Err(FleetIpcRankInstallError::DescriptorCount {
            expected: keys.len(),
            actual: descriptors.len(),
        });
    }
    for (ordinal, (&key, descriptor)) in keys.iter().zip(descriptors).enumerate() {
        if descriptor.key() != key {
            return Err(FleetIpcRankInstallError::DescriptorKeyMismatch(
                ordinal as u64,
            ));
        }
    }
    Ok(())
}

fn descriptor_digest(
    plan_identity: [u8; 32],
    proof_generation: u64,
    install_domain: IpcExchangeInstallDomain,
    descriptors: &[IpcExchangeDescriptor],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DESCRIPTOR_DIGEST_TAG);
    hasher.update(&plan_identity);
    hasher.update(&proof_generation.to_le_bytes());
    hasher.update(install_domain.as_bytes());
    hasher.update(&(descriptors.len() as u64).to_le_bytes());
    for descriptor in descriptors {
        hasher.update(&descriptor.encode());
    }
    *hasher.finalize().as_bytes()
}
