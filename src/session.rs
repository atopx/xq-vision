use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

use crate::config::ModelSource;
use crate::config::SessionConfig;
use crate::error::Result;
use crate::error::XqVisionError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderFailure {
    Fallback,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Cuda,
    TensorRt,
    DirectMl,
    OpenVino,
    Xnnpack,
}

impl ExecutionProvider {
    fn dispatch(self, failure: ProviderFailure) -> Result<ExecutionProviderDispatch> {
        let dispatch = match self {
            Self::Cpu => ort::execution_providers::CPUExecutionProvider::default().build(),
            Self::CoreMl => {
                #[cfg(feature = "coreml")]
                {
                    ort::execution_providers::CoreMLExecutionProvider::default().build()
                }
                #[cfg(not(feature = "coreml"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    ort::execution_providers::CUDAExecutionProvider::default().build()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
            Self::TensorRt => {
                #[cfg(feature = "tensorrt")]
                {
                    ort::execution_providers::TensorRTExecutionProvider::default().build()
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
            Self::DirectMl => {
                #[cfg(feature = "directml")]
                {
                    ort::execution_providers::DirectMLExecutionProvider::default().build()
                }
                #[cfg(not(feature = "directml"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
            Self::OpenVino => {
                #[cfg(feature = "openvino")]
                {
                    ort::execution_providers::OpenVINOExecutionProvider::default().build()
                }
                #[cfg(not(feature = "openvino"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
            Self::Xnnpack => {
                #[cfg(feature = "xnnpack")]
                {
                    ort::execution_providers::XNNPACKExecutionProvider::default().build()
                }
                #[cfg(not(feature = "xnnpack"))]
                {
                    return Err(XqVisionError::UnsupportedProvider { provider: self });
                }
            }
        };

        Ok(match failure {
            ProviderFailure::Fallback => dispatch.fail_silently(),
            ProviderFailure::Error => dispatch.error_on_failure(),
        })
    }
}

pub(crate) fn create_session(source: &ModelSource, config: &SessionConfig) -> Result<Session> {
    let mut builder =
        Session::builder()?.with_optimization_level(config.graph_optimization().into()).map_err(map_builder_error)?;

    if let Some(threads) = config.intra_threads() {
        builder = builder.with_intra_threads(threads).map_err(map_builder_error)?;
    }
    if let Some(threads) = config.inter_threads() {
        builder = builder.with_inter_threads(threads).map_err(map_builder_error)?;
    }
    if config.parallel_execution() {
        builder = builder.with_parallel_execution(true).map_err(map_builder_error)?;
    }

    let providers = config
        .execution_providers()
        .iter()
        .copied()
        .map(|provider| provider.dispatch(config.provider_failure()))
        .collect::<Result<Vec<_>>>()?;
    builder = builder.with_execution_providers(providers).map_err(map_builder_error)?;

    match source {
        ModelSource::File(path) => Ok(builder.commit_from_file(path)?),
        ModelSource::Memory(bytes) => Ok(builder.commit_from_memory(bytes)?),
    }
}

fn map_builder_error(error: ort::Error<SessionBuilder>) -> XqVisionError {
    XqVisionError::Ort(ort::Error::new(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_provider_dispatches_without_feature_gate() -> Result<()> {
        let dispatch = ExecutionProvider::Cpu.dispatch(ProviderFailure::Fallback)?;
        assert!(dispatch.downcast_ref::<ort::execution_providers::CPUExecutionProvider>().is_some());
        Ok(())
    }
}
