#![allow(warnings)]

mod config;
mod cuda;
mod vanity;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::VanityConfig;
use env_logger::Builder;
use log::{info, LevelFilter};
use std::time::Instant;
use vanity::VanityGenerator;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Prefixes to search for, comma separated
    #[arg(short, long, value_delimiter = ',')]
    prefixes: Option<Vec<String>>,

    /// Suffixes to search for, comma separated
    #[arg(short, long, value_delimiter = ',')]
    suffixes: Option<Vec<String>>,

    /// Case insensitive matching
    #[arg(short, long, default_value_t = true)]
    ignore_case: bool,

    /// Use GPU for computation
    #[arg(short = 'g', long, default_value_t = true)]
    gpu: bool,

    /// Number of CPU threads to use
    #[arg(short, long, default_value_t = 8)]
    threads: usize,

    /// Batch size for each iteration
    #[arg(short, long, default_value_t = 100000)]
    batch_size: usize,

    /// Output file to save found keys
    #[arg(short, long, default_value = "found_keys.txt")]
    output: String,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logger
    let mut builder = Builder::new();
    builder.filter_level(if cli.verbose {
        LevelFilter::Debug
    } else {
        LevelFilter::Info
    });
    builder.init();

    // Build configuration
    let mut config = VanityConfig::new();

    if let Some(prefixes) = cli.prefixes {
        config = config.with_prefixes(prefixes);
    }

    if let Some(suffixes) = cli.suffixes {
        config = config.with_suffixes(suffixes);
    }

    config = config
        .with_ignore_case(cli.ignore_case)
        .with_gpu_enabled(cli.gpu)
        .with_threads(cli.threads)
        .with_batch_size(cli.batch_size)
        .with_output_file(Some(cli.output));

    // Print starting information
    info!("Rust Vanity Address Generator");
    info!("----------------------------");
    info!("Using GPU: {}", config.gpu_enabled);
    info!("Threads: {}", config.threads);
    info!("Batch size: {}", config.batch_size);
    if !config.prefixes.is_empty() {
        info!("Prefixes: {:?}", config.prefixes);
    }
    if !config.suffixes.is_empty() {
        info!("Suffixes: {:?}", config.suffixes);
    }
    info!("Case insensitive: {}", config.ignore_case);
    info!("Output file: {:?}", config.output_file);
    info!("----------------------------");

    // Create vanity generator
    let mut generator = VanityGenerator::new(config)?;

    // Start timer
    let start = Instant::now();

    // Run the generator
    generator.run()?;

    // Print elapsed time
    let elapsed = start.elapsed();
    info!(
        "Finished in {:.2} seconds",
        elapsed.as_secs_f64()
    );

    Ok(())
}
