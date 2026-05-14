use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use ort::session::builder::GraphOptimizationLevel;

use crate::session::ExecutionProvider;
use crate::session::ProviderFailure;

#[derive(Debug, Clone)]
pub enum ModelSource {
    File(PathBuf),
    Memory(Arc<[u8]>),
}

impl ModelSource {
    #[must_use]
    pub fn file(path: impl Into<PathBuf>) -> Self { Self::File(path.into()) }

    #[must_use]
    pub fn memory(bytes: impl Into<Vec<u8>>) -> Self { Self::Memory(Arc::from(bytes.into())) }

    #[must_use]
    pub fn as_file(&self) -> Option<&Path> {
        match self {
            Self::File(path) => Some(path.as_path()),
            Self::Memory(_) => None,
        }
    }

    #[must_use]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::File(_) => None,
            Self::Memory(bytes) => Some(bytes),
        }
    }
}

impl From<PathBuf> for ModelSource {
    fn from(value: PathBuf) -> Self { Self::file(value) }
}

impl From<&Path> for ModelSource {
    fn from(value: &Path) -> Self { Self::file(value) }
}

impl From<&str> for ModelSource {
    fn from(value: &str) -> Self { Self::file(value) }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphOptimization {
    Disable,
    Basic,
    Extended,
    Layout,
    #[default]
    All,
}

impl From<GraphOptimization> for GraphOptimizationLevel {
    fn from(value: GraphOptimization) -> Self {
        match value {
            GraphOptimization::Disable => Self::Disable,
            GraphOptimization::Basic => Self::Level1,
            GraphOptimization::Extended => Self::Level2,
            GraphOptimization::Layout => Self::Level3,
            GraphOptimization::All => Self::All,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    execution_providers: Vec<ExecutionProvider>,
    provider_failure: ProviderFailure,
    graph_optimization: GraphOptimization,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    parallel_execution: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            execution_providers: default_execution_providers(),
            provider_failure: ProviderFailure::Fallback,
            graph_optimization: GraphOptimization::All,
            intra_threads: None,
            inter_threads: None,
            parallel_execution: false,
        }
    }
}

/// Build the default provider list from compiled-in execution-provider features.
///
/// Accelerators come first (in a stable order) and `Cpu` is always appended as
/// the final fallback. When no provider feature is enabled the list is just
/// `[Cpu]`. The provider list is intentionally not configurable at runtime —
/// the cargo feature set is the single source of truth.
fn default_execution_providers() -> Vec<ExecutionProvider> {
    vec![
        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda,
        #[cfg(feature = "tensorrt")]
        ExecutionProvider::TensorRt,
        #[cfg(feature = "coreml")]
        ExecutionProvider::CoreMl,
        #[cfg(feature = "directml")]
        ExecutionProvider::DirectMl,
        #[cfg(feature = "openvino")]
        ExecutionProvider::OpenVino,
        #[cfg(feature = "xnnpack")]
        ExecutionProvider::Xnnpack,
        ExecutionProvider::Cpu,
    ]
}

impl SessionConfig {
    #[must_use]
    pub fn new() -> Self { Self::default() }

    #[must_use]
    pub fn execution_providers(&self) -> &[ExecutionProvider] { &self.execution_providers }

    #[must_use]
    pub fn provider_failure(&self) -> ProviderFailure { self.provider_failure }

    #[must_use]
    pub fn graph_optimization(&self) -> GraphOptimization { self.graph_optimization }

    #[must_use]
    pub fn intra_threads(&self) -> Option<usize> { self.intra_threads }

    #[must_use]
    pub fn inter_threads(&self) -> Option<usize> { self.inter_threads }

    #[must_use]
    pub fn parallel_execution(&self) -> bool { self.parallel_execution }

    #[must_use]
    pub fn with_provider_failure(mut self, failure: ProviderFailure) -> Self {
        self.provider_failure = failure;
        self
    }

    #[must_use]
    pub fn with_graph_optimization(mut self, level: GraphOptimization) -> Self {
        self.graph_optimization = level;
        self
    }

    #[must_use]
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    #[must_use]
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    #[must_use]
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }
}
