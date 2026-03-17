mod cli;

use clap::{Parser, Subcommand};

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
    },
    /// Load model with tiered scheduling and run inference
    Run {
        /// Path to model file
        model: String,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
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
        /// Path to model file
        model: String,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
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
        Commands::Estimate { model } => cli::estimate::run(&model),
        Commands::Run {
            model,
            context,
            prompt,
            interactive,
            max_tokens,
        } => cli::run::run(&model, context, prompt.as_deref(), interactive, max_tokens),
        Commands::Serve { model, host, port, context } => cli::serve::run(&model, &host, port, context),
        Commands::Bench {
            model,
            baseline,
            context,
            max_tokens,
            prompt,
            force,
        } => cli::bench::run(&model, baseline, context, max_tokens, prompt.as_deref(), force),
        Commands::Inspect { model, tensors } => cli::inspect::run(&model, tensors),
        Commands::Optimize { model } => cli::optimize::run(&model),
    }
}
