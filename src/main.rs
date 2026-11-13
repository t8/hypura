mod cli;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hypura", version, about = "Storage-tier-aware LLM inference scheduler")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hardware profiler and save results
    Profile {
        /// Force re-profiling even if a recent profile exists
        #[arg(long)]
        force: bool,
    },
    /// Show performance estimate for a model without loading it
    Estimate {
        /// Path to model file or HuggingFace model ID
        model: String,
        /// Maximum context length to estimate placement and KV cache for
        #[arg(short = 'c', long = "context", visible_alias = "ctx-size", default_value = "8192")]
        context: u32,
    },
    /// List all available local and Ollama models
    List {
        /// Custom models directory to scan
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// Ollama models directory (default: ~/.ollama/models)
        #[arg(long)]
        ollama_models: Option<PathBuf>,
    },
    /// Pull a model from Hugging Face or registry into the models directory
    Pull {
        /// Model name, tag, or Hugging Face repo (e.g. gemma-4:26b, qwen3.8:27b, owner/repo:quant)
        model: String,
        /// Target models directory (defaults to configured models directory or /Volumes/RealtekData/models)
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// Overwrite existing file if it already exists
        #[arg(long)]
        force: bool,
    },
    /// Remove/delete a model and free disk space
    #[command(alias = "delete")]
    Rm {
        /// Model name, tag, or file path to remove
        model: String,
        /// Target models directory
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
    /// List currently running and active loaded models
    Ps {
        /// Host of the running Hypura server
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port of the running Hypura server
        #[arg(long, default_value = "6000")]
        port: u16,
    },
    /// Load model with tiered scheduling and run inference
    Run {
        /// Path to model file
        model: String,
        /// Maximum context length
        #[arg(short = 'c', long = "context", visible_alias = "ctx-size", default_value = "4096")]
        context: u32,
        /// Single prompt (non-interactive mode)
        #[arg(long)]
        prompt: Option<String>,
        /// Interactive chat mode
        #[arg(long)]
        interactive: bool,
        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,
    },
    /// Start Ollama-compatible API server
    Serve {
        /// Optional initial model to pre-load (dynamic on-demand if omitted)
        model: Option<String>,
        /// Optional initial model name or path via flag
        #[arg(short = 'm', long = "model")]
        model_flag: Option<String>,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(short = 'p', long = "port", default_value = "8080")]
        port: u16,
        /// Default context length (can be overridden per request via num_ctx)
        #[arg(short = 'c', long = "context", visible_alias = "ctx-size", default_value = "4096")]
        context: u32,
        /// Custom models directory to scan
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// Ollama models directory (default: ~/.ollama/models)
        #[arg(long)]
        ollama_models: Option<PathBuf>,
        /// Idle timeout in seconds before unloading model to free unified memory (0 to disable)
        #[arg(long, default_value = "0")]
        idle_timeout: u64,
        /// KV cache quantization type: f16, q8_0, or q4_0 (auto-selected by context size if omitted)
        #[arg(long)]
        kv_quant: Option<String>,
    },
    /// Benchmark tok/s: Hypura scheduling vs naive mmap
    Bench {
        /// Path to model file
        model: String,
        /// Also benchmark with naive mmap for comparison
        #[arg(long)]
        baseline: bool,
        /// Maximum context length
        #[arg(long, default_value = "2048")]
        context: u32,
        /// Tokens to generate per run
        #[arg(long, default_value = "128")]
        max_tokens: u32,
        /// Prompt text
        #[arg(long)]
        prompt: Option<String>,
        /// Force unsafe operations (e.g. baseline with model larger than RAM)
        #[arg(long)]
        force: bool,
    },
    /// Print model metadata, tensor list, and placement plan
    Inspect {
        /// Path to model file
        model: String,
        /// Show individual tensor details
        #[arg(long)]
        tensors: bool,
    },
    /// Low-level NVMe I/O microbenchmark (diagnostic)
    Iobench {
        /// Path to a GGUF model file
        model: String,
        /// Amount of data to read in each test (GiB)
        #[arg(long, default_value = "1.0")]
        read_gb: f64,
    },
    /// (MoE only) Reorganize expert layout on disk for sequential access
    Optimize {
        /// Path to model file
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("HYPURA_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Profile { force } => cli::profile::run(force),
        Commands::Estimate { model, context } => cli::estimate::run(&model, context),
        Commands::List {
            models_dir,
            ollama_models,
        } => cli::list::run(models_dir, ollama_models),
        Commands::Pull {
            model,
            models_dir,
            force,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cli::pull::run(model, models_dir, force))
        }
        Commands::Rm { model, models_dir } => cli::rm::run(model, models_dir),
        Commands::Ps { host, port } => cli::ps::run(&host, port),
        Commands::Run {
            model,
            context,
            prompt,
            interactive,
            max_tokens,
        } => cli::run::run(&model, context, prompt.as_deref(), interactive, max_tokens),
        Commands::Serve {
            model,
            model_flag,
            host,
            port,
            context,
            models_dir,
            ollama_models,
            idle_timeout,
            kv_quant,
        } => cli::serve::run(
            model.or(model_flag).as_deref(),
            &host,
            port,
            context,
            models_dir,
            ollama_models,
            idle_timeout,
            kv_quant,
        ),
        Commands::Bench {
            model,
            baseline,
            context,
            max_tokens,
            prompt,
            force,
        } => cli::bench::run(&model, baseline, context, max_tokens, prompt.as_deref(), force),
        Commands::Inspect { model, tensors } => cli::inspect::run(&model, tensors),
        Commands::Iobench { model, read_gb } => cli::iobench::run(&model, read_gb),
        Commands::Optimize { model } => cli::optimize::run(&model),
    }
}
